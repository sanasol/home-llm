"""Bounded Orca IQ4_NL -> native Sushi Engram FP8 adapter (lossy requantization).

OCP E4M3FN RNE, one E8M0 power-of-two scale per 32 columns. This is
NOT original FP8 recovery. Native Table.open requires weight/scale in one shard.
CPU only; no torch or GPU is needed outside the optional independent oracle.
Convention: deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/kernel.py act_quant
(round_scale=True, block32); E4M3FN round-to-nearest-even, no BF16 intermediate.
Consumer: Sushi d5e876e dsv41_engram.zig Table.open/dequant; runtime BF16
rounding occurs AFTER the table gather/dequant, before the wkv linear.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from orca_export import atomic_json, digest

LAYERS = {1: (0, 384006168), 14: (384006168, 384016682)}
DONOR = "per_layer_token_embd.weight"
CONVENTION = "e4m3fn-rne-e8m0-row32-amax-floor1e-4-v1"
# Positive finite E4M3FN, increasing code order; max finite code is 0x7e.
CODES = np.arange(127, dtype=np.uint32)
EXP, MAN = CODES >> 3, CODES & 7
VALUES = np.where(EXP == 0, MAN * 2.0**-9,
                  (8 + MAN) * np.exp2(EXP.astype(np.int32) - 10)).astype(np.float32)


def quantize_rows(rows):
    """F32 donor values -> U8 E4M3 codes and U8 E8M0 exponent codes.

    Scale follows public inference/kernel.py act_quant(round_scale=True), with
    block32 and max(amax,1e-4); precision floor is explicit, not a lossless claim.
    No intermediate BF16 cast. No NaN sentinel or clipping overflow is accepted.
    """
    x = np.asarray(rows, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % 32 or not np.isfinite(x).all():
        raise ValueError("Expected finite F32 [rows,columns] with columns multiple32")
    blocks = x.reshape(x.shape[0], -1, 32)
    amax = np.maximum(np.abs(blocks).max(-1), np.float32(1e-4))
    ratio = amax * np.float32(1 / 448)
    # frexp avoids log2 rounding a value just above a power of two down to it.
    mantissa, exponent = np.frexp(ratio)
    power = exponent - (mantissa == np.float32(0.5))
    if np.any((power < -126) | (power > 127)):
        raise ValueError("Scale outside normal finite E8M0/F32 range")
    scale = np.ldexp(np.ones_like(amax), power)
    normalized = blocks / scale[..., None]
    a = np.abs(normalized)
    if np.any(a > 448):
        raise ValueError("Scale calculation overflow")
    high = np.minimum(np.searchsorted(VALUES, a, side="left"), 126)
    low = np.maximum(high - 1, 0)
    dl, dh = a - VALUES[low], VALUES[high] - a
    code = np.where((dl < dh) | ((dl == dh) & ((low & 1) == 0)), low, high).astype(np.uint8)
    code |= np.where(np.signbit(normalized), 128, 0).astype(np.uint8)
    return code.reshape(x.shape), (power + 127).astype(np.uint8)


def dequantize_rows(codes, scales):
    codes, scales = np.asarray(codes), np.asarray(scales)
    if codes.dtype != np.uint8 or scales.dtype != np.uint8 or codes.ndim != 2 or scales.shape != (codes.shape[0], codes.shape[1] // 32):
        raise ValueError("Invalid native Engram dtype/geometry")
    if np.any((codes & 127) == 127) or np.any(scales == 255):
        raise ValueError("Nonfinite native Engram code")
    values = VALUES[codes & 127] * np.where(codes & 128, -1, 1).astype(np.float32)
    return values * np.repeat(np.ldexp(np.ones(scales.shape, np.float32), scales.astype(np.int32) - 127), 32, axis=1)


def range_digest(file, offset, size):
    h = hashlib.sha256()
    file.seek(offset)
    while size:
        block = file.read(min(size, 8 << 20))
        if not block:
            raise ValueError("Truncated Engram output")
        h.update(block)
        size -= len(block)
    return h.hexdigest()


def export_table(reader, output, layer, *, chunk_rows=8192, row_start=0, row_count=None, backend="cpu"):
    """Single-writer, row-chunk resumable shard; sample exports stamped as samples.

    Two bounded random writes/chunk co-locate complete weight and scale arrays.
    Progress commits only after fsync. Resume verifies every committed byte range.
    """
    from orca_reader import REPO, REVISION
    if backend not in ("cpu", "cuda"):
        raise ValueError("Engram backend must be cpu or cuda")
    if layer not in LAYERS or type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("Supported Engram layers1/14; positive chunk_rows required")
    first, total = LAYERS[layer]
    rows = total - row_start if row_count is None else row_count
    if type(row_start) is not int or type(rows) is not int or not 0 <= row_start < total or not 0 < rows <= total - row_start:
        raise ValueError("Invalid native row slice")
    t = reader.tensors[DONOR]
    if t.quant != "IQ4_NL" or t.shape != (sum(v[1] for v in LAYERS.values()), 256):
        raise ValueError("Pinned donor Engram geometry/quant mismatch")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    state_path = output.with_suffix(output.suffix + ".resume.json")
    pin = {"repo": REPO, "revision": REVISION, "layer": layer, "native_start": row_start,
           "donor_start": first + row_start, "rows": rows, "columns": 256, "chunk_rows": chunk_rows,
           "convention": CONVENTION, "backend": backend, "source": vars(t),
           "code_sha256": {name: digest(Path(__file__).parent / name)
                           for name in ("engram_export.py", "engram_cuda.py", "orca_reader.py", "orca_export.py", "hub.py")}}
    # JSON roundtrip canonicalizes tuple-valued Tensor.dims before comparison.
    pin = json.loads(json.dumps(pin))
    metadata = {"adapter": CONVENTION, "source_repo": REPO, "source_revision": REVISION,
                "conversion": "lossy-donor-IQ4_NL-dequant-F32-to-native-FP8;not-original-FP8",
                "native_row_start": str(row_start), "status": "full-table" if rows == total and row_start == 0 else "sample-not-loadable-full-table"}
    prefix = f"layers.{layer}.engram.embed."
    weight_size, scale_size = rows * 256, rows * 8
    header = {"__metadata__": metadata,
              prefix + "weight": {"dtype": "F8_E4M3", "shape": [rows, 256], "data_offsets": [0, weight_size]},
              prefix + "scale": {"dtype": "F8_E8M0", "shape": [rows, 8], "data_offsets": [weight_size, weight_size + scale_size]}}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    start = 8 + len(raw)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["pin"] != pin:
            raise ValueError("Engram resume pin mismatch")
        recovered_rename = not state.get("done") and not partial.exists() and output.exists()
        if recovered_rename and sum(c["rows"] for c in state["chunks"]) != rows:
            raise ValueError("Final Engram output exists before all chunks completed")
        existing = output if state.get("done") or recovered_rename else partial
        with existing.open("rb") as f:
            if f.read(start) != struct.pack("<Q", len(raw)) + raw:
                raise ValueError("Engram output header changed")
            for record in state["chunks"]:
                r, count = record["row"], record["rows"]
                for leaf, offset, size in (("weight", start + r * 256, count * 256),
                                           ("scale", start + weight_size + r * 8, count * 8)):
                    if range_digest(f, offset, size) != record[leaf]:
                        raise ValueError("Engram committed chunk changed")
        if recovered_rename:
            state["done"] = True
            state["output_bytes"] = output.stat().st_size
            state["relative_frobenius"] = float(np.sqrt(state["sse"] / max(state["signal_sq"], 1e-300)))
            atomic_json(state_path, state)
        if state.get("done"):
            return state
    else:
        if output.exists() or partial.exists():
            raise ValueError("Refusing to overwrite untracked Engram output")
        state = {"pin": pin, "chunks": [], "done": False, "sse": 0.0, "signal_sq": 0.0}
        with partial.open("xb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw)
            f.truncate(start + weight_size + scale_size)
            f.flush()
            os.fsync(f.fileno())
        atomic_json(state_path, state)
    next_row = sum(c["rows"] for c in state["chunks"])
    with partial.open("r+b") as f:
        for r in range(next_row, rows, chunk_rows):
            count = min(chunk_rows, rows - r)
            dense = reader.read(DONOR, row_start=first + row_start + r, row_stop=first + row_start + r + count)
            if backend == "cuda":
                from engram_cuda import quantize_rows as cuda_quantize
                codes, scales, metrics = cuda_quantize(dense, metrics=True)
            else:
                codes, scales = quantize_rows(dense)
                decoded = dequantize_rows(codes, scales)
                error = dense.astype(np.float64) - decoded
                metrics = {"sse":float(np.square(error).sum()),
                           "signal_sq":float(np.square(dense.astype(np.float64)).sum())}
            record = {"row": r, "rows": count, "weight": hashlib.sha256(codes.tobytes()).hexdigest(),
                      "scale": hashlib.sha256(scales.tobytes()).hexdigest()}
            for a, offset in ((codes, start + r * 256), (scales, start + weight_size + r * 8)):
                f.seek(offset)
                a.tofile(f)
            f.flush()
            os.fsync(f.fileno())
            state["chunks"].append(record)
            state["sse"] += metrics["sse"]
            state["signal_sq"] += metrics["signal_sq"]
            atomic_json(state_path, state)
    from pair1_contract import read_header
    read_header(partial)
    # Both absolute paths are the exact sibling paths validated/constructed above.
    if partial.resolve() != output.with_suffix(output.suffix + ".partial") or output.exists():
        raise ValueError("Unsafe or existing final Engram target")
    os.replace(partial, output)
    state["done"] = True
    state["output_bytes"] = output.stat().st_size
    state["relative_frobenius"] = float(np.sqrt(state["sse"] / max(state["signal_sq"], 1e-300)))
    atomic_json(state_path, state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--layer", type=int, choices=LAYERS, required=True)
    p.add_argument("--catalog")
    p.add_argument("--local-source-dir", help="Verified local GGUF cache root with exact HF relative paths")
    p.add_argument("--chunk-rows", type=int, default=8192)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--row-count", type=int)
    a = p.parse_args()
    if a.gpu < 0:
        p.error("GPU must be nonnegative")
    if a.backend == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        os.environ["OMP_NUM_THREADS"] = "1"
    from orca_reader import OrcaReader
    reader = (OrcaReader(a.catalog, local_source_dir=a.local_source_dir) if a.catalog
              else OrcaReader(local_source_dir=a.local_source_dir))
    import fcntl
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Engram output already owned by another converter")
        state = export_table(reader, a.output, a.layer, chunk_rows=a.chunk_rows,
                             row_start=a.row_start, row_count=a.row_count, backend=a.backend)
    print(json.dumps({k: v for k, v in state.items() if k not in ("pin", "chunks")}, indent=2))


if __name__ == "__main__":
    main()
