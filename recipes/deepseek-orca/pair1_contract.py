"""Independent PAIR1 pack oracle. No encoder imports; NumPy + public Sushi fixtures.

Decoded tiles are row-major [..,256], matrices are [in,out]. Validation is
structural, not proof of quantization quality or donor identity.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

# Frozen public consumer rows: Sushi expert_exl3.zig, PAIR1_ROWS (n24..64).
PAIR1_ROWS = (
    (0x58F41BF3, 3, 0x8ED78C13, 0x8EEE8C03),
    (0x58F41BF3, 3, 0x8ED78C13, 0x8EEE8C03),
    (0x7AA2BE15, 3, 0x8ED78C13, 0x8ECE8C03),
    (0x7B229E11, 3, 0x8ED78C13, 0x8ECE8C03),
    (0x3B22BE15, 3, 0x8ED78C13, 0x8ECE8C03),
    (0x46C63E17, 19, 0x8F9A8E6E, 0x8E258FFE),
    (0x46C63E17, 19, 0x8F9A8E6E, 0x8E258FFE),
    (0x46C63E17, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0x46C63E17, 19, 0x8F9A8E6E, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0x82439F67, 19, 0x8E6E8F9A, 0x8E258FAE),
    (0xE2C7FC0D, 19, 0x8F408FB0, 0x8F918F89),
    (0xE2C7FC0D, 19, 0x8F408FA0, 0x8F118FC9),
    (0xE2C7FC0D, 19, 0x8F408FA0, 0x8F118FC9),
    (0xE2C7FC0D, 19, 0x8F408FA0, 0x8F118FC9),
)
TABLES = {
    "pair1-w16-v1": ("pair1", "c089b4eb", 24, 64),
    "spec1-w15": ("spec1", "37f72431", 16, 128),
    "spec1-w16": ("spec1", "753ffa03", 24, 64),
    "std1": ("std1", "c5dba171", 24, 64),
}
PAIR1_CONFIG = dict(format="stq", k=4, codebook="pair1", window=16,
                    codebook_table="pair1-w16-v1", codebook_digest="c089b4eb")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def row(n):
    _require(type(n) is int and 24 <= n <= 64 and n % 2 == 0, "PAIR1 rate must be even n24..64")
    return PAIR1_ROWS[(n - 24) // 2]


def table_digest():
    text = "".join(str(n) + "".join(f",{x:08x}" for x in row(n)) + "\n" for n in range(24, 65, 2))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:8]


def pair1_values(codewords, n):
    """Odd-step uint16 windows -> [...,2] f16 values, including half-sum rounding."""
    mult, rot, mask0, mask1 = row(n)
    cw = np.asarray(codewords)
    _require(cw.dtype.kind in "iu" and np.all((cw >= 0) & (cw <= 65535)), "invalid codeword")
    p = (cw.astype(np.uint64) * mult & 0xFFFFFFFF).astype(np.uint32)
    rotated = (p << rot) | (p >> (32 - rot))
    result = []
    for mixed, mask in ((p, mask0), (rotated, mask1)):
        bits = mixed & mask
        lo = (bits & 65535).astype(np.uint16).view(np.float16).astype(np.float32)
        hi = (bits >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
        result.append(((lo + hi).astype(np.float16).astype(np.float32) * 4096).astype(np.float16))
    return np.stack(result, axis=-1)


def unpack_codes(packed):
    """Read circular 16-bit windows via explicit MSB-first bits, any admitted n."""
    packed = np.asarray(packed)
    _require(packed.ndim >= 1 and packed.dtype.kind == "u" and packed.dtype.itemsize == 2, "trellis must be uint16")
    n = packed.shape[-1]
    row(n)
    words = packed[..., ::2].astype(np.uint32) | (packed[..., 1::2].astype(np.uint32) << 16)
    bits = ((words[..., None] >> np.arange(31, -1, -1, dtype=np.uint32)) & 1).reshape(*packed.shape[:-1], 16 * n)
    ends = ((np.arange(256) + 1) * n) // 16
    indices = (ends[:, None] - 16 + np.arange(16)) % (16 * n)
    return np.sum(bits[..., indices] * (1 << np.arange(15, -1, -1)), axis=-1).astype(np.uint16)


def core_perm():
    """Trellis step -> row-major position inside one input-by-output 16x16 tile."""
    return np.array([16 * (2 * (t % 4) + r) + t // 4 + c
                     for t in range(32) for c in (0, 8) for r in (0, 1, 8, 9)])


def decode_tiles(packed, chunk_tiles=512):
    """Decode row-major tiles with bounded bit-window temporaries, not expert-sized ones."""
    packed = np.asarray(packed)
    _require(packed.ndim >= 1 and packed.dtype.kind == "u" and packed.dtype.itemsize == 2, "trellis must be uint16")
    _require(type(chunk_tiles) is int and chunk_tiles > 0, "chunk_tiles must be positive")
    n = packed.shape[-1]
    row(n)
    flat = packed.reshape(-1, n)
    out = np.empty((len(flat), 256), dtype=np.float16)
    perm = core_perm()
    for start in range(0, len(flat), chunk_tiles):
        codes = unpack_codes(flat[start:start + chunk_tiles])
        values = pair1_values(codes[:, 1::2], n).reshape(-1, 256)
        out[start:start + chunk_tiles, perm] = values
    return out.reshape(*packed.shape[:-1], 256)


def inner_from_trellis(packed, tile_order="pack"):
    """[... ,IT,OT,n] or [...,OT/2,IT,2,n] -> [...,in,out], unscaled."""
    packed = np.asarray(packed)
    if tile_order == "pair_major":
        _require(packed.ndim >= 4 and packed.shape[-2] == 2, "invalid pair-major shape")
        packed = np.swapaxes(packed, -4, -3).reshape(*packed.shape[:-4], packed.shape[-3], 2 * packed.shape[-4], packed.shape[-1])
    else:
        _require(tile_order == "pack" and packed.ndim >= 3, "invalid pack order")
    it, ot = packed.shape[-3:-1]
    tiles = decode_tiles(packed).reshape(*packed.shape[:-3], it, ot, 16, 16)
    return np.swapaxes(tiles, -3, -2).reshape(*packed.shape[:-3], it * 16, ot * 16)


def reconstruct(packed, suh, svh, tile_order="pack"):
    """Ideal f32 [in,out] weight; excludes activation/output rounding in runtime."""
    inner = inner_from_trellis(packed, tile_order).astype(np.float32)
    su, sv = np.asarray(suh), np.asarray(svh)
    _require(su.dtype == np.float16 and sv.dtype == np.float16, "scales must be F16")
    _require(su.shape == inner.shape[:-1] and sv.shape == inner.shape[:-2] + inner.shape[-1:], "scale geometry")
    i, o = inner.shape[-2:]
    _require(i % 128 == o % 128 == 0, "Hadamard requires 128 grid")
    ids = np.arange(128)
    h = np.array([[(-1) ** int(a & b).bit_count() for b in ids] for a in ids], np.float32) / np.sqrt(np.float32(128))
    x = (h @ inner.reshape(*inner.shape[:-2], i // 128, 128, o)).reshape(inner.shape)
    x = (x.reshape(*x.shape[:-1], o // 128, 128) @ h).reshape(inner.shape)
    return su.astype(np.float32)[..., :, None] * x * sv.astype(np.float32)[..., None, :]


def validate_quant(block, expected=None):
    _require(isinstance(block, dict) and block.get("format") == "stq", "STQ quant block required")
    table = block.get("codebook_table")
    _require(isinstance(table, str) and table in TABLES, "unknown codebook table")
    cb, digest, low, high = TABLES[table]
    _require(block.get("codebook") == cb and (expected is None or cb == expected), "codebook/table mismatch")
    _require(block.get("codebook_digest") == digest, "codebook digest mismatch (legacy SPEC1 digest is retired)")
    window = block.get("window", 16)
    _require(type(window) is int and 8 <= window <= 16 and (cb != "pair1" or window == 16), "codebook window mismatch")
    k = block.get("k")
    _require(type(k) in (int, float) and math.isfinite(k), "invalid k")
    n = k * 16
    _require(n == int(n) and int(n) % 2 == 0 and low <= n <= high, "ceiling rate missing from table")
    _require("tile_order" not in block or block["tile_order"] == "pair_major", "unsupported tile order")
    return int(n)


DTYPES = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "U16": 2, "I16": 2,
          "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8, "BOOL": 1}


def read_header(path):
    """Validate safetensors coverage and bounds without reading weight payloads."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        _require(len(prefix) == 8, "truncated safetensors prefix")
        length = int.from_bytes(prefix, "little")
        _require(2 <= length <= min(size - 8, 64 << 20), "invalid safetensors header length")
        def unique(pairs):
            obj = {}
            for k, v in pairs:
                _require(k not in obj, "duplicate safetensors key")
                obj[k] = v
            return obj
        header = json.loads(f.read(length), object_pairs_hook=unique)
    _require(isinstance(header, dict), "header must be object")
    metadata = header.get("__metadata__", {})
    _require(isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()), "metadata must contain strings")
    intervals = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        _require(isinstance(info, dict), f"invalid tensor {name}")
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        _require(dtype in DTYPES and isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape), f"invalid dtype/shape {name}")
        _require(isinstance(offsets, list) and len(offsets) == 2 and all(type(d) is int for d in offsets), f"invalid offsets {name}")
        start, end = offsets
        _require(0 <= start <= end <= size - 8 - length and end - start == math.prod(shape) * DTYPES[dtype], f"tensor byte geometry {name}")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        _require(start == cursor, "safetensors overlap or unrepresented gap")
        cursor = end
    _require(cursor == size - 8 - length, "unrepresented trailing bytes")
    return header


def read_tensors(path):
    """Small public fixtures only; production validation uses read_header instead."""
    header = read_header(path)
    with Path(path).open("rb") as f:
        start = 8 + int.from_bytes(f.read(8), "little")
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            _require(info["dtype"] in ("U16", "F16", "F32", "U8"), "fixture dtype unsupported")
            a, b = info["data_offsets"]
            f.seek(start + a)
            dtype = {"U16": "<u2", "F16": "<f2", "F32": "<f4", "U8": "u1"}[info["dtype"]]
            tensors[name] = np.frombuffer(f.read(b - a), dtype=dtype).reshape(info["shape"])
    return tensors


ROUTED = re.compile(r"^(model\.layers\.(\d+)|mtp\.(\d+))\.mlp\.switch_mlp\.(gate|up|down)_proj(?:\.g(\d+))?\.(trellis_pm|trellis|suh|svh)$")


def validate_routed(tensors, quant, hidden, intermediate, router_rows, top_k=6):
    """Validate header entries for ONE complete layer/stage; return group sizes/rates.

    router_rows is checked against total experts. Router permutation itself needs the
    export manifest's source IDs, not only headers; see validate_permutation.
    """
    ceiling = validate_quant(quant, "pair1")
    _require(type(hidden) is int and type(intermediate) is int and hidden > 0 and intermediate > 0 and hidden % 128 == intermediate % 128 == 0, "invalid routed dimensions")
    banks, layers, grouped = {}, set(), set()
    for key, info in tensors.items():
        m = ROUTED.fullmatch(key)
        if not m:
            _require(".mlp.switch_mlp." not in key, f"invalid routed name {key}")
            continue
        prefix, _, _, projection, group, part = m.groups()
        layers.add(prefix)
        grouped.add(group is not None)
        g = int(group or 0)
        _require(group is None or str(g) == group, "noncanonical group name")
        banks.setdefault((g, projection), {})[part] = info
    _require(len(layers) == 1 and len(grouped) == 1 and banks, "missing/mixed routed layer layout")
    groups = sorted({g for g, _ in banks})
    _require(groups == list(range(len(groups))) and len(groups) <= 32, "missing/invalid group")
    pair_major = quant.get("tile_order") == "pair_major"
    _require(not pair_major or grouped == {True}, "pair-major requires rate groups")
    result = []
    for g in groups:
        counts, rates = [], []
        for projection in ("gate", "up", "down"):
            parts = banks.get((g, projection), {})
            leaf = "trellis_pm" if pair_major else "trellis"
            _require(set(parts) == {leaf, "suh", "svh"}, "incomplete/mixed-order routed bank")
            tr, su, sv = (parts[p] for p in (leaf, "suh", "svh"))
            i, o = (intermediate, hidden) if projection == "down" else (hidden, intermediate)
            shape = tr["shape"]
            _require(len(shape) == (5 if pair_major else 4), "routed trellis rank")
            e, n = shape[0], shape[-1]
            row(n)
            expected = [e, o // 32, i // 16, 2, n] if pair_major else [e, i // 16, o // 16, n]
            _require(e > 0 and shape == expected and n <= ceiling, "routed trellis geometry/rate")
            _require(tr["dtype"] == "U16" and su["dtype"] == sv["dtype"] == "F16", "routed dtype")
            _require(su["shape"] == [e, i] and sv["shape"] == [e, o], "routed scales geometry")
            counts.append(e)
            rates.append(n)
        _require(len(set(counts)) == 1, "projection expert counts disagree")
        result.append({"group": g, "experts": counts[0], "n": rates})
    total = sum(g["experts"] for g in result)
    _require(type(router_rows) is int and total == router_rows and type(top_k) is int and 0 < top_k <= min(total, 32), "router rows/top-k mismatch")
    return result


def validate_permutation(source_ids, experts):
    """New concatenated group row -> original expert id; full Orca export never prunes."""
    _require(type(experts) is int and experts > 0 and len(source_ids) == experts and
             all(type(i) is int for i in source_ids) and sorted(source_ids) == list(range(experts)), "invalid expert permutation")
    return np.asarray(source_ids, dtype=np.int64)


def validate_stamp(metadata, quant, revision=None):
    """Strict writer contract; deliberately stronger than DeepSeek's tile-only check."""
    ceiling = validate_quant(quant)
    _require(isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values()), "metadata values must be strings")
    _require(metadata.get("format") == "stq" and metadata.get("codebook") == quant["codebook"], "shard codebook stamp mismatch")
    try:
        k, window = float(metadata["k"]), int(metadata["window"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("invalid shard rate/window stamp") from exc
    low, high = TABLES[quant["codebook_table"]][2:]
    _require(math.isfinite(k) and k * 16 == int(k * 16) and int(k * 16) % 2 == 0 and low <= k * 16 <= min(high, ceiling) and window == quant.get("window", 16), "shard rate/window mismatch")
    _require(metadata.get("tile_order") == quant.get("tile_order"), "shard tile-order mismatch")
    if revision is not None:
        _require(metadata.get("pack_revision") == revision, "pack revision mismatch")


def validate_trunk(tensors, quant):
    ceiling = validate_quant(quant, "spec1")
    _require("tile_order" not in quant, "trunk trellises stay pack-order")
    low, high = TABLES[quant["codebook_table"]][2:]
    count = 0
    for name, tr in tensors.items():
        if not name.endswith(".trellis") or ROUTED.fullmatch(name):
            continue
        base = name[:-8]
        _require(not any(base + suffix in tensors for suffix in (".weight", ".scale", ".scales", ".biases")), "STQ/dense duplicate trunk")
        su, sv = tensors.get(base + ".suh"), tensors.get(base + ".svh")
        shape = tr["shape"]
        _require(len(shape) in (3, 4) and (len(shape) == 3 or base.endswith(".attn.wo_a")), "trunk bank rank")
        _require(all(type(d) is int and d > 0 for d in shape), "trunk dimensions")
        it, ot, n = shape[-3:]
        prefix = shape[:-3]
        _require(tr["dtype"] == "U16" and su is not None and sv is not None and su["dtype"] == sv["dtype"] == "F16", "trunk bank dtype/scales")
        _require(it % 8 == ot % 8 == 0 and low <= n <= min(high, ceiling) and n % 2 == 0, "trunk bank grid/rate")
        _require(su["shape"] == prefix + [16 * it] and sv["shape"] == prefix + [16 * ot], "trunk scale geometry")
        count += 1
    return count
