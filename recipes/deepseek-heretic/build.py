"""Abliterated trunk overlay for the official Sushi DeepSeek-V4.1-Flash-1017 STQ 1.75bpw pack.

The official pack's routed experts remain unchanged. Only 38 Heretic attention wo_b writers (layers2-39) are re-encoded into
the pack's own SPEC1 trellis banks (same rate, same scales, same tensor names/shapes), so the shard layout is unchanged.

  build.py slab   <donor> <layer> [leaf]      CPU smoke test on the first 128 input columns of one tensor
  build.py encode <donor> [--part i/n]        encode every edited tensor (GPU), resumable, writes metrics.jsonl
  build.py assemble <donor>                   patch the official trunk shards -> $DS_WORK/pack/<donor>/
"""
import argparse, json, os, sys, time, tempfile
import numpy as np, torch
from hub import rng, st_header, st_tensor, api
import stq

BASE = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
OFF = "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw"
OFF_REV = "89630812557263bb7a7582fb1f30455573511155"
WORK = os.path.abspath(os.environ.get("DS_WORK", "./heretic-work"))
HERE = os.path.dirname(os.path.abspath(__file__))
LEAVES = {"wo_b": "attn.wo_b", "w2": "ffn.shared_experts.w2"}
BASE_REV = "08c8657f02291e11b6f6b63e0e16014f8e748518"
DONOR_REV = "12d58ad0878b5008e80d720251adb969a0069412"
DONORS = {"heretic": dict(repo="pqhaz/DeepSeek-V4.1-Flash-1017-heretic", path="patch/heretic_patch.safetensors", leaves=["wo_b"])}

_cache = {}


def base_index():
    if "idx" in _cache:
        return _cache["idx"]
    p = os.path.join(WORK, f"base_index.{BASE_REV}.json")
    mapping = None
    if os.path.exists(p):
        try:
            with open(p) as source:
                mapping = json.load(source)["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Empty weight_map")
        except (ValueError, KeyError):
            mapping = None  # Repair old partial caches only with a validated atomic replacement.
    if mapping is None:
        raw = rng(BASE, "model.safetensors.index.json", 0, 50_000_000, BASE_REV)
        mapping = json.loads(raw)["weight_map"]
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Pinned base index has no weight_map")
        os.makedirs(WORK, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=WORK, prefix=".base-index-", delete=False) as out:
                temporary = out.name
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, p)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
    _cache["idx"] = mapping
    return mapping


def hdr(repo, shard, rev):
    key = (repo, shard, rev)
    if key not in _cache:
        _cache[key] = st_header(repo, shard, rev)
    return _cache[key]


def fp8_raw(repo, name, shard=None):
    shard = shard or base_index()[name + ".weight"]
    w, wi = st_tensor(repo, shard, name + ".weight", hdr(repo, shard, BASE_REV), BASE_REV)
    s, si = st_tensor(repo, shard, name + ".scale", hdr(repo, shard, BASE_REV), BASE_REV)
    assert wi["dtype"] == "F8_E4M3" and si["dtype"] == "F8_E8M0", (name, wi["dtype"], si["dtype"])
    return np.frombuffer(w, np.uint8).reshape(wi["shape"]), np.frombuffer(s, np.uint8).reshape(si["shape"])


def official_bank(layer, name):
    shard = f"model-trunk-{layer:02d}.safetensors"
    out = []
    for leaf, dt in (("trellis", np.uint16), ("suh", np.float16), ("svh", np.float16)):
        b, info = st_tensor(OFF, shard, f"{name}.{leaf}", hdr(OFF, shard, OFF_REV), OFF_REV)
        out.append(np.frombuffer(b, dt).reshape(info["shape"]))
    return out


def donor_dense(donor, name, base_codes, base_scale):
    """The donor's weight as f32 [out, in], or None where the donor left the tensor as the base has it."""
    d = DONORS[donor]
    h = hdr(d["repo"], d["path"], DONOR_REV)
    if name + ".weight" not in h[0]:
        raise ValueError("Pinned patch is missing expected edited tensor: " + name)
    b, info = st_tensor(d["repo"], d["path"], name + ".weight", h, DONOR_REV)
    if info["dtype"] != "F8_E4M3" or info["shape"] != list(base_codes.shape):
        raise ValueError("Unsupported patch tensor: " + name)
    codes, scale = np.frombuffer(b, np.uint8).reshape(info["shape"]), base_scale
    if name + ".scale" in h[0]:
        raise ValueError("Unexpected donor scale tensor")
    if codes.shape == base_codes.shape and (codes == base_codes).all() and (scale == base_scale).all():
        return None
    return stq.fp8_dense(codes, scale)


def coef(w, r, ref):
    """Signed share of the base's component along r that w keeps (w, ref as [in, out], r in output space)."""
    a, b = w @ r, ref @ r
    return float((a * b).sum() / (b * b).sum())


def encode_tensor(target_io, tr, su, sv, dev, batch, in_blocks=None):
    """target [in, out] torch on dev -> (new trellis U16 like tr, reconstruction [in, out])."""
    it, ot, n = tr.shape
    x = stq.tiles_of(stq.target_inner(target_io, su, sv))
    if in_blocks is not None:
        x = x[: in_blocks * 8 * ot]
    codes, _ = stq.encode_tiles(x, n, batch)
    new = tr.copy()
    new.reshape(-1, n)[: codes.shape[0]] = codes
    return new, stq.reconstruct(new, su, sv, dev)


def process(donor, layer, leaf, dev, batch, in_blocks=None):
    name = f"layers.{layer}.{LEAVES[leaf]}"
    bc, bs = fp8_raw(BASE, name)
    dd = donor_dense(donor, name, bc, bs)
    if dd is None:
        return None, {"tensor": name, "donor": donor, "skipped": "identical to base"}
    t0 = time.time()
    tr, su, sv = official_bank(layer, name)
    base = torch.from_numpy(stq.fp8_dense(bc, bs)).T.contiguous().to(dev)       # [in, out]
    tgt = torch.from_numpy(dd).T.contiguous().to(dev)
    new, rec = encode_tensor(tgt, tr, su, sv, dev, batch, in_blocks)
    off = stq.reconstruct(tr, su, sv, dev)
    rows = slice(0, in_blocks * 128) if in_blocks else slice(None)
    base, tgt, rec, off = base[rows], tgt[rows], rec[rows], off[rows]
    delta = tgt - base
    r = torch.svd_lowrank(delta.T, q=2, niter=6)[0][:, 0]                        # edit direction, output space
    rel = lambda a, b: float((a - b).norm() / b.norm())
    m = {"tensor": name, "donor": donor, "n": int(tr.shape[-1]), "tiles": int(tr.shape[0] * tr.shape[1]) if not in_blocks else in_blocks * 8 * tr.shape[1],
         "edit_rel": rel(tgt, base), "ours_vs_donor": rel(rec, tgt), "official_vs_base": rel(off, base),
         "dir_coef_donor": coef(tgt, r, base), "dir_coef_ours": coef(rec, r, base), "dir_coef_official": coef(off, r, base),
         "edit_retained": float(((rec - off) * delta).sum() / (delta * delta).sum()),
         "codes_changed": float((new != tr).any(-1).mean()), "sec": round(time.time() - t0, 1)}
    return new, m


def tensors_of(donor):
    return [(l, leaf) for l in range(2, 40) for leaf in DONORS[donor]["leaves"]]


def cmd_encode(a):
    if not torch.cuda.is_available():
        raise RuntimeError("Full encoding requires CUDA; use slab for CPU smoke")
    dev = "cuda"
    out = os.path.join(WORK, "out", a.donor)
    os.makedirs(out, exist_ok=True)
    i, n = (int(x) for x in a.part.split("/"))
    if not 0 <= i < n:
        raise ValueError("Part must satisfy 0 <= i < n")
    for j, (layer, leaf) in enumerate(tensors_of(a.donor)):
        f = os.path.join(out, f"layers.{layer}.{LEAVES[leaf]}.npy")
        if layer % n != i or os.path.exists(f) or os.path.exists(f + ".skip"):
            continue
        new, m = process(a.donor, layer, leaf, dev, a.batch)
        if new is None:
            open(f + ".skip", "w").close()
        else:
            np.save(f + ".tmp.npy", new)
            os.replace(f + ".tmp.npy", f)
        with open(os.path.join(out, f"metrics.{i}.jsonl"), "a") as fh:
            fh.write(json.dumps(m) + "\n")
        print(json.dumps(m), flush=True)
    print(f"ENCODE COMPLETE {a.donor} part {a.part}", flush=True)


def cmd_assemble(a):
    """Official trunk shard with the re-encoded trellis bytes written over the same offsets (header untouched)."""
    out, pack = os.path.join(WORK, "out", a.donor), os.path.join(WORK, "pack", a.donor)
    os.makedirs(pack, exist_ok=True)
    missing = [f"layers.{layer}.{LEAVES[leaf]}" for layer, leaf in tensors_of(a.donor)
               if not any(os.path.isfile(os.path.join(out, f"layers.{layer}.{LEAVES[leaf]}.npy" + suffix))
                          for suffix in ("", ".skip"))]
    if missing:
        raise RuntimeError(f"Incomplete encoding; missing {len(missing)} tensors: {missing}")
    total = 0
    for layer in range(40):
        got = {LEAVES[l]: os.path.join(out, f"layers.{layer}.{LEAVES[l]}.npy") for l in DONORS[a.donor]["leaves"]}
        got = {k: v for k, v in got.items() if os.path.exists(v)}
        if not got:
            continue
        shard = f"model-trunk-{layer:02d}.safetensors"
        cached = os.path.join(WORK, "official-trunk", OFF_REV, shard)
        if os.path.isfile(cached):
            import struct
            buf = bytearray(open(cached, "rb").read())
            off = 8 + struct.unpack_from("<Q", buf)[0]
            h = json.loads(buf[8:off])
            h.pop("__metadata__", None)
        else:
            h, off, _ = hdr(OFF, shard, OFF_REV)
            size = off + max(v["data_offsets"][1] for v in h.values())
            buf = bytearray(rng(OFF, shard, 0, size - 1, OFF_REV))
        size = off + max(v["data_offsets"][1] for v in h.values())
        assert len(buf) == size, (shard, len(buf), size)
        for leaf, f in got.items():
            info = h[f"layers.{layer}.{leaf}.trellis"]
            new = np.load(f)
            assert info["dtype"] == "U16" and list(new.shape) == info["shape"], (f, new.shape, info)
            s, e = info["data_offsets"]
            assert e - s == new.nbytes
            buf[off + s: off + e] = new.astype("<u2").tobytes()
            total += 1
        open(os.path.join(pack, shard), "wb").write(buf)
    print(f"ASSEMBLE COMPLETE {a.donor}: {total} tensors in {len(os.listdir(pack))} shards -> {pack}")


def cmd_slab(a):
    torch.set_num_threads(6)
    _, m = process(a.donor, a.layer, a.leaf, "cpu", 256, in_blocks=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slab"); s.add_argument("donor", choices=("heretic",)); s.add_argument("layer", type=int); s.add_argument("leaf", nargs="?", default="wo_b"); s.set_defaults(f=cmd_slab)
    s = sub.add_parser("encode"); s.add_argument("donor", choices=("heretic",)); s.add_argument("--part", default="0/1"); s.add_argument("--batch", type=int, default=4096); s.set_defaults(f=cmd_encode)
    s = sub.add_parser("assemble"); s.add_argument("donor", choices=("heretic",)); s.set_defaults(f=cmd_assemble)
    a = p.parse_args()
    a.f(a)
