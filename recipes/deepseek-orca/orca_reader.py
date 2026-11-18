"""Pinned, bounded Orca Mix-256G tensor reads. No base-weight substitution.

GGUF dims are fastest-axis first; read() returns ordinary C-order [out,in]
(or [expert,out,in]). Sushi trunk matrices use [in,out]: transpose explicitly.
Install/use the work venv's gguf package; its IQ tables and MXFP4 codec are the
reference rather than approximate home-grown decoders.
"""
import hashlib
import json
import math
import os
import operator
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as QT, GGML_QUANT_SIZES
from gguf.quants import dequantize
from hub import api, gguf_header, rng

REPO = "orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF"
REVISION = "83144a667924660c6ba0095585fd039d8daa9b91"
CATALOG = Path(__file__).with_name("orca_256g_tensors.json")
VISION = "mmproj-DeepSeek-V4.1-Flash-1017-abliterated-BF16.gguf"


def shard_path(shard):
    return f"Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-{shard:05d}-of-00007.gguf"


@dataclass(frozen=True)
class Tensor:
    path: str
    dims: tuple
    quant: str
    offset: int
    data_start: int

    @property
    def shape(self):
        return tuple(reversed(self.dims))

    @property
    def nbytes(self):
        block, size = GGML_QUANT_SIZES[QT[self.quant]]
        if not self.dims or any(d <= 0 for d in self.dims) or self.dims[0] % block:
            raise ValueError(f"Invalid {self.quant} tensor dimensions: {self.dims}")
        return math.prod(self.dims) // block * size

    @property
    def row_bytes(self):
        block, size = GGML_QUANT_SIZES[QT[self.quant]]
        return self.dims[0] // block * size


# Every non-fused text tensor in the pinned Mix-256G catalog.
LAYER_MAP = {
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "exp_probs_b_vl.bias": "ffn.gate.bias_vl",
    "engram_kv.weight": "engram.wkv.weight",
    **{f"ffn_{g}_shexp.weight": f"ffn.shared_experts.w{w}.weight"
       for g, w in (("gate", 1), ("down", 2), ("up", 3))},
    **{f"{n}.weight": n for n in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                                                   "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale")},
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
}
GLOBAL_MAP = {"output.weight": "head.weight", "output_norm.weight": "norm.weight",
              "token_embd.weight": "embed.weight", "mm.1.weight": "aligner.w1.weight",
              "mm.1.bias": "aligner.w1.bias", "mm.2.weight": "aligner.w2.weight",
              "mm.2.bias": "aligner.w2.bias", "v.post_ln.weight": "vision.norm.weight",
              "v.patch_embd.weight": "vision.patch_embed.proj.weight",
              "v.patch_embd.bias": "vision.patch_embed.proj.bias",
              "v.token_embd.img_start": "image_start", "v.token_embd.img_end": "image_end",
              "v.image_newline": "image_newline"}
VISION_MAP = {"attn_out": "attn.wo", "attn_qkv": "attn.wqkv", "ln1": "norm1",
              "ln2": "norm2", "ffn_down": "mlp.w2"}


def native_names(name, tensor):
    """Native destinations; composite tensors return multiple names, never drop edits."""
    if name in GLOBAL_MAP:
        return (GLOBAL_MAP[name],)
    if name == "per_layer_token_embd.weight":
        return ("layers.1.engram.embed.weight", "layers.14.engram.embed.weight")
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if m:
        layer, tail = m.groups()
        prefix = f"layers.{layer}."
        if tail in LAYER_MAP:
            return (prefix + LAYER_MAP[tail],)
        e = re.fullmatch(r"ffn_(gate|down|up)_exps.weight", tail)
        if e:
            w = {"gate": 1, "down": 2, "up": 3}[e[1]]
            return tuple(prefix + f"ffn.experts.{i}.w{w}.weight" for i in range(tensor.shape[0]))
        if tail == "engram_gate.weight":
            # Converter collapses q/k into one gate; not an invertible rename.
            return (prefix + "engram.q_weight", prefix + "engram.k_weight")
    m = re.fullmatch(r"v.blk\.(\d+)\.(.+)\.(weight|bias)", name)
    if m:
        layer, tail, suffix = m.groups()
        native = "mlp.w1" if tail in ("ffn_gate", "ffn_up") else VISION_MAP[tail]
        return (f"vision.blocks.{layer}.{native}.{suffix}",)
    raise KeyError(f"Unmapped donor tensor: {name}")


class OrcaReader:
    def __init__(self, catalog=CATALOG, *, fetch=rng, max_range_bytes=8 << 20,
                 local_source_dir=None, metadata_fetch=api):
        self.fetch = fetch
        self.local_source_dir = Path(local_source_dir).resolve() if local_source_dir is not None else None
        self.metadata_fetch = metadata_fetch
        self._local_files = {}
        self._file_metadata = None
        self.max_range_bytes = operator.index(max_range_bytes)
        if self.max_range_bytes <= 0:
            raise ValueError("max_range_bytes must be positive")
        if not isinstance(catalog, dict):
            with open(catalog) as f:
                catalog = json.load(f)
        self.tensors = {}
        self._verified = set()
        for name, (shard, dims, quant, offset, end, align) in catalog.items():
            if align <= 0 or align & (align - 1) or not 1 <= shard <= 7 or offset < 0:
                raise ValueError("Invalid GGUF catalog")
            t = Tensor(shard_path(shard), tuple(dims), quant, offset, (end + align - 1) // align * align)
            t.nbytes  # Validate every format and block width before any network call.
            self.tensors[name] = t

    def _local_file(self, path):
        if self.local_source_dir is None:
            return None
        local = (self.local_source_dir / path).resolve()
        if not local.is_relative_to(self.local_source_dir):
            raise ValueError("Local source path escapes cache directory")
        if not local.exists():
            if path in self._local_files:
                raise ValueError("Verified local source disappeared")
            return None
        if path in self._local_files:
            f, fingerprint, provenance = self._local_files[path]
            if self._fingerprint(os.fstat(f.fileno())) != fingerprint or self._fingerprint(local.stat()) != fingerprint:
                raise ValueError("Verified local source changed")
            return f
        if self._file_metadata is None:
            info = self.metadata_fetch(f"models/{REPO}/revision/{REVISION}?blobs=true")
            if info.get("sha") != REVISION:
                raise ValueError("Local source metadata revision mismatch")
            self._file_metadata = {x["rfilename"]: x for x in info["siblings"]}
        info = self._file_metadata.get(path, {})
        lfs = info.get("lfs") or {}
        expected_hash = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:")
        expected_size = info.get("size", lfs.get("size"))
        if not re.fullmatch(r"[a-f0-9]{64}", expected_hash) or not isinstance(expected_size, int):
            raise ValueError("Pinned LFS SHA256 and size required for local GGUF")
        f = local.open("rb")
        try:
            before = self._fingerprint(os.fstat(f.fileno()))
            if before[2] != expected_size:
                raise ValueError("Local GGUF size mismatch")
            digest = hashlib.sha256()
            while block := f.read(8 << 20):
                digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError("Local GGUF SHA256 mismatch")
            if self._fingerprint(os.fstat(f.fileno())) != before:
                raise ValueError("Local GGUF changed during verification")
            provenance = {"repo": REPO, "revision": REVISION, "path": path,
                          "size": expected_size, "sha256": expected_hash, "verification": "full_LFS_SHA256"}
            self._local_files[path] = (f, before, provenance)
            return f
        except Exception:
            f.close()
            raise

    @staticmethod
    def _fingerprint(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def local_provenance(self):
        return {path: entry[2].copy() for path, entry in self._local_files.items()}

    def close(self):
        for f, _, _ in self._local_files.values():
            f.close()
        self._local_files.clear()

    def _bytes(self, path, start, size):
        if start < 0 or size < 0:
            raise ValueError("Invalid byte range")
        local = self._local_file(path)
        if local is not None:
            data = os.pread(local.fileno(), size, start)
            if len(data) != size:
                raise EOFError("Short verified local GGUF read")
            return data
        parts = []
        for pos in range(start, start + size, self.max_range_bytes):
            n = min(self.max_range_bytes, start + size - pos)
            for attempt in range(5):
                data = self.fetch(REPO, path, pos, pos + n - 1, REVISION)
                if len(data) == n:
                    break
                if attempt == 4:
                    raise EOFError(f"Short range read: expected {n}, got {len(data)}")
            parts.append(data)
        return b"".join(parts)

    def _header(self, path, size=None):
        size = size or 65536
        while size <= 16 << 20:
            buf = self._bytes(path, 0, size)
            try:
                ts, kv, end = gguf_header(buf)
                break
            except (EOFError, struct.error):
                size *= 2
        else:
            raise ValueError("GGUF header exceeds 16 MiB limit")
        if buf[:8] != b"GGUF\x03\x00\x00\x00":
            raise ValueError("Expected little-endian GGUF v3")
        align = kv.get("general.alignment", 32)
        if align <= 0 or align & (align - 1):
            raise ValueError("Invalid GGUF alignment")
        data_start = (end + align - 1) // align * align
        tensors = {n: Tensor(path, tuple(d), QT(q).name, off, data_start)
                   for n, (d, q, off) in ts.items()}
        ordered = sorted(tensors.values(), key=lambda t: t.offset)
        for i, t in enumerate(ordered):
            if t.offset < 0 or t.offset % align:
                raise ValueError("Invalid tensor offset")
            if i + 1 < len(ordered) and t.offset + t.nbytes > ordered[i + 1].offset:
                raise ValueError("Overlapping GGUF tensors")
        return tensors

    def verify(self, path):
        if path not in self._verified:
            expected = {n: t for n, t in self.tensors.items() if t.path == path}
            size = max((t.data_start for t in expected.values()), default=None)
            actual = self._header(path, size)
            if expected and expected != actual:
                raise ValueError("Catalog does not match pinned GGUF header")
            self.tensors.update(actual)
            self._verified.add(path)

    def add_vision(self):
        self.verify(VISION)
        return self

    def read(self, name, *, expert=None, row_start=0, row_stop=None):
        """Read contiguous flattened outer rows; expert selects one fused MoE matrix.

        A rank-one tensor is one row. A partial read always returns [rows,width];
        a full read restores the original reversed-GGUF shape.
        """
        t = self.tensors[name]
        self.verify(t.path)
        shape = t.shape
        first = 0
        if expert is not None:
            expert = operator.index(expert)
            if len(shape) != 3 or not 0 <= expert < shape[0] or "_exps." not in name:
                raise ValueError("Invalid routed expert selection")
            first = expert * shape[1]
            shape = shape[1:]
        rows = math.prod(shape[:-1])
        row_start = operator.index(row_start)
        row_stop = rows if row_stop is None else operator.index(row_stop)
        if not 0 <= row_start <= row_stop <= rows:
            raise ValueError("Invalid row interval")
        raw = self._bytes(t.path, t.data_start + t.offset + (first + row_start) * t.row_bytes,
                          (row_stop - row_start) * t.row_bytes)
        if not raw:
            return np.empty((0, shape[-1]), np.float32)
        codes = np.frombuffer(raw, np.uint8).reshape(row_stop - row_start, t.row_bytes)
        values = dequantize(codes, QT[t.quant]).astype(np.float32, copy=False)
        return values.reshape(shape if row_start == 0 and row_stop == rows else (row_stop - row_start, shape[-1]))

    def iter_rows(self, name, *, expert=None, rows_per_chunk=128):
        rows_per_chunk = operator.index(rows_per_chunk)
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        shape = self.tensors[name].shape
        rows = math.prod((shape[1:] if expert is not None else shape)[:-1])
        for start in range(0, rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, rows)
            yield start, self.read(name, expert=expert, row_start=start, row_stop=stop).reshape(stop - start, shape[-1])

    def native_sources(self):
        """Native name -> donor names. Vision is included after add_vision()."""
        out = {}
        for name, t in self.tensors.items():
            for native in native_names(name, t):
                out.setdefault(native, []).append(name)
        return out

    def read_native(self, name, *, row_start=0, row_stop=None, sushi=False):
        """Native [out,in] rows, or transposed Sushi [in,out] chunk.

        Engram q/k factorization cannot be recovered from the fused donor gate:
        fail closed rather than silently restoring unedited base weights.
        """
        sources = self.native_sources()[name]
        donor = sources[0]
        if donor.endswith("engram_gate.weight"):
            raise ValueError("Fused engram q/k gate is not invertible; retain the donor gate explicitly")
        if name.endswith("engram.embed.weight"):
            boundary = 384006168
            offset = 0 if name.startswith("layers.1.") else boundary
            total = boundary if offset == 0 else 384016682
            stop = total if row_stop is None else row_stop
            if not 0 <= row_start <= stop <= total:
                raise ValueError("Invalid engram embedding rows")
            result = self.read(donor, row_start=offset + row_start, row_stop=offset + stop)
        elif ".mlp.w1.weight" in name:
            # Native SwiGLU w1 is [gate;up], not interleaved or summed.
            sources = sorted(sources, key=lambda n: "ffn_up" in n)
            half = self.tensors[sources[0]].shape[0]
            stop = 2 * half if row_stop is None else row_stop
            if not 0 <= row_start <= stop <= 2 * half:
                raise ValueError("Invalid fused vision rows")
            pieces = [self.read(s, row_start=max(0, row_start - i * half),
                                row_stop=min(half, stop - i * half))
                      for i, s in enumerate(sources) if max(row_start, i * half) < min(stop, (i + 1) * half)]
            result = np.concatenate(pieces) if pieces else np.empty((0, self.tensors[donor].shape[-1]), np.float32)
        elif name == "vision.patch_embed.proj.weight":
            t = self.tensors[donor]
            total, channels, height, width = t.shape
            stop = total if row_stop is None else row_stop
            if not 0 <= row_start <= stop <= total:
                raise ValueError("Invalid patch embedding rows")
            result = self.read(donor, row_start=row_start * channels * height,
                               row_stop=stop * channels * height).reshape(stop - row_start, channels * height * width)
        else:
            expert = int(name.split(".experts.")[1].split(".")[0]) if ".experts." in name else None
            result = self.read(donor, expert=expert, row_start=row_start, row_stop=row_stop)
        if sushi:
            if result.ndim != 2:
                raise ValueError("Sushi linear transpose requires a matrix")
            result = result.T.copy()
        return result
