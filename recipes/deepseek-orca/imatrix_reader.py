"""Lazy pinned Orca activation diagonals in native input coordinates, not STQ space.

Use raw sums or common layer normalization when ranking different experts; dividing
by each expert's own count erases routing frequency. No dense covariance is built.
"""
import hashlib
import math
import operator
import re
from dataclasses import dataclass

import numpy as np
from hub import gguf_header, rng
from orca_reader import LAYER_MAP, REPO, REVISION, OrcaReader

PATH = "imatrix.gguf"


@dataclass(frozen=True)
class Importance:
    raw_sum2: np.ndarray
    count: float
    layer_count: float
    provenance: dict

    def weights(self, normalization="layer"):
        if normalization not in ("raw", "layer", "expert"):
            raise ValueError("normalization must be raw, layer, or expert")
        denominator = {"raw": 1.0, "layer": self.layer_count, "expert": self.count}[normalization]
        return self.raw_sum2.astype(np.float64) / denominator


def donor_name(native):
    m = re.fullmatch(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight", native)
    if m:
        layer, expert, projection = map(int, m.groups())
        if not 0 <= layer < 40 or not 0 <= expert < 384:
            raise ValueError("Invalid routed layer/expert")
        projection = {1: "gate", 2: "down", 3: "up"}[projection]
        return f"blk.{layer}.ffn_{projection}_exps.weight", expert
    m = re.fullmatch(r"layers\.(\d+)\.(.+)", native)
    if not m or not 0 <= int(m[1]) < 40:
        raise KeyError(f"No calibrated donor mapping: {native}")
    reverse = {v: k for k, v in LAYER_MAP.items()}
    if m[2] not in reverse:
        raise KeyError(f"No calibrated donor mapping: {native}")
    return f"blk.{m[1]}.{reverse[m[2]]}", None


class ImatrixReader:
    def __init__(self, *, fetch=rng):
        self._reader = OrcaReader({}, fetch=fetch, max_range_bytes=262144)
        self._header = None
        self._counts = {}

    def _load(self):
        if self._header is not None:
            return
        raw = self._reader._bytes(PATH, 0, 262144)
        tensors, metadata, end = gguf_header(raw)
        if raw[:8] != b"GGUF\x03\x00\x00\x00" or metadata.get("general.type") != "imatrix":
            raise ValueError("Expected pinned GGUF v3 imatrix")
        alignment = metadata.get("general.alignment", 32)
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Invalid importance alignment")
        start = (end + alignment - 1) // alignment * alignment
        ordered = sorted(tensors.items(), key=lambda item: item[1][2])
        for i, (name, (dims, quant, offset)) in enumerate(ordered):
            if quant != 0 or not dims or any(d <= 0 for d in dims) or offset < 0 or offset % alignment:
                raise ValueError(f"Invalid F32 importance tensor: {name}")
            if i + 1 < len(ordered) and offset + math.prod(dims) * 4 > ordered[i + 1][1][2]:
                raise ValueError("Overlapping importance tensors")
        layer_count = metadata.get("imatrix.chunk_count", 0) * metadata.get("imatrix.chunk_size", 0)
        if not math.isfinite(layer_count) or layer_count <= 0:
            raise ValueError("Missing positive calibration token count")
        self.data_start = start
        self.layer_count = float(layer_count)
        self.header_sha256 = hashlib.sha256(raw[:end]).hexdigest()
        self._header = tensors

    def identity(self):
        """Pinned source/header identity; fetches only the bounded header once."""
        self._load()
        return {"repo": REPO, "revision": REVISION, "path": PATH,
                "header_sha256": self.header_sha256, "layer_count": self.layer_count,
                "coordinates": "native_input", "ranking_normalization": "raw_or_common_layer"}

    def coverage(self):
        """Scan only 120 routed count vectors, never importance or weight payloads."""
        from concurrent.futures import ThreadPoolExecutor
        self._load()
        keys = [f"blk.{layer}.ffn_{proj}_exps.weight.counts"
                for layer in range(40) for proj in ("gate", "down", "up")]
        def inspect(key):
            if key not in self._header:
                return key, {"status": "missing"}
            if self._header[key][0] != [1, 384]:
                return key, {"status": "invalid_shape"}
            raw = self._range(key, 0, 384)
            values = np.frombuffer(raw, "<f4")
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > self.layer_count):
                return key, {"status": "invalid_counts"}
            self._counts[key] = raw
            return key, {"status": "verified", "zero_experts": np.flatnonzero(values == 0).tolist(),
                         "min": float(values.min()), "max": float(values.max()),
                         "sha256": hashlib.sha256(raw).hexdigest(),
                         "byte_range": [self.data_start + self._header[key][2],
                                        self.data_start + self._header[key][2] + len(raw) - 1]}
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = dict(pool.map(inspect, keys))
        return {"source": self.identity(), "projection_vectors": len(records),
                "zero_projection_experts": sum(len(v.get("zero_experts", [])) for v in records.values()),
                "unverified_vectors": [k for k, v in records.items() if v["status"] != "verified"],
                "records": records}

    def _range(self, key, first, size):
        dims, _, offset = self._header[key]
        if not 0 <= first <= first + size <= math.prod(dims):
            raise ValueError("Importance range exceeds tensor")
        return self._reader._bytes(PATH, self.data_start + offset + first * 4, size * 4)

    def read_native(self, native_name, *, group=None):
        """Fetch one native input diagonal; wo_a requires explicit output group 0..7.

        Group g applies to wo_a native output rows [g*1024:(g+1)*1024].
        For ordinary projections group must be omitted. Zero-count/unvisited or
        entirely zero/nonfinite/negative diagonals fail closed; individual inactive
        input channels may be zero. No invented calibration.
        """
        donor, expert = donor_name(native_name)
        self._load()
        key, count_key = donor + ".in_sum2", donor + ".counts"
        if key not in self._header or count_key not in self._header:
            raise KeyError(f"No importance calibration: {native_name}")
        dims = self._header[key][0]
        expected_dims = [dims[0], 384] if expert is not None else [dims[0]]
        expected_counts = [1, 384] if expert is not None else [1]
        if dims != expected_dims or self._header[count_key][0] != expected_counts:
            raise ValueError("Unexpected calibrated expert/input shape")
        width = dims[0]
        first = (expert or 0) * width
        if donor.endswith("attn_output_a.weight"):
            if group is None:
                raise ValueError("Grouped wo_a requires group=0..7; diagonal is not shared across groups")
            group = operator.index(group)
            if not 0 <= group < 8 or width != 32768:
                raise ValueError("Invalid wo_a calibration group")
            width //= 8
            first += group * width
        elif group is not None:
            raise ValueError("group applies only to wo_a")
        if count_key not in self._counts:
            self._counts[count_key] = self._range(count_key, 0, math.prod(expected_counts))
        count_raw = self._counts[count_key][(expert or 0) * 4:((expert or 0) + 1) * 4]
        count = float(np.frombuffer(count_raw, "<f4")[0])
        raw = self._range(key, first, width)
        values = np.frombuffer(raw, "<f4").copy()
        if not math.isfinite(count) or not 0 < count <= self.layer_count:
            raise ValueError("Unvisited expert or invalid calibration count")
        if not np.isfinite(values).all() or np.any(values < 0) or not np.any(values > 0):
            raise ValueError("Importance diagonal must be finite, nonnegative, and not entirely zero")
        values.flags.writeable = False
        provenance = {"repo": REPO, "revision": REVISION, "path": PATH,
                      "header_sha256": self.header_sha256, "native_name": native_name,
                      "sum2_tensor": key, "expert": expert, "group": group,
                      "sum2_byte_range": [self.data_start + self._header[key][2] + first * 4,
                                          self.data_start + self._header[key][2] + (first + width) * 4 - 1],
                      "sum2_sha256": hashlib.sha256(raw).hexdigest(),
                      "count_sha256": hashlib.sha256(count_raw).hexdigest(),
                      "sample_sha256": hashlib.sha256(raw + count_raw).hexdigest(),
                      "count": count, "layer_count": self.layer_count,
                      "coordinates": "native_input", "ranking_normalization": "raw_or_common_layer"}
        return Importance(values, count, self.layer_count, provenance)
