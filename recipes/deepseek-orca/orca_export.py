"""Pilot-first Orca GGUF -> Sushi exporter. No official weights or inferred edits.

A projection pilot is an artifact, not a loadable model. A layer pilot includes
new PAIR1 experts and lossless F32 trunk; it is NOT the final 1.75bpw pack.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

RATES = (24, 28, 32, 40)
PROJECTIONS = {"gate": "w1", "up": "w3", "down": "w2"}
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with path.with_suffix(path.suffix + ".tmp").open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path.with_suffix(path.suffix + ".tmp"), path)


class Manifest:
    """Single-writer work directory; completed bytes are verified on every resume."""
    def __init__(self, directory, pin):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["pin"] != pin:
                raise ValueError("Resume pin mismatch; use a different work directory")
        else:
            self.data = {"version": 1, "pin": pin, "complete": {}}
            atomic_json(self.path, self.data)

    def ready(self, key):
        record = self.data["complete"].get(key)
        if record is None:
            return False
        path = self.root / record["file"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Completed artifact changed or missing: {key}")
        return True

    def finish(self, key, filename, metrics=None):
        path = self.root / filename
        if path.parent.resolve() != self.root.resolve():
            raise ValueError("Artifacts must be direct children of work directory")
        self.data["complete"][key] = {"file": filename, "sha256": digest(path),
                                      "bytes": path.stat().st_size, "metrics": metrics or {}}
        atomic_json(self.path, self.data)


def write_safetensors(path, arrays, metadata, dtype_overrides=None):
    """Small work items only; final layer assembly streams their byte ranges."""
    dtypes = {np.dtype("uint16"): "U16", np.dtype("float16"): "F16", np.dtype("float32"): "F32"}
    arrays = {k: np.ascontiguousarray(v, dtype=v.dtype.newbyteorder("<")) for k, v in arrays.items()}
    header, offset = {"__metadata__": metadata}, 0
    for name, a in arrays.items():
        stored_dtype = (dtype_overrides or {}).get(name, dtypes[a.dtype])
        if stored_dtype != dtypes[a.dtype] and not (stored_dtype == "BF16" and a.dtype == np.uint16):
            raise ValueError("Only BF16-on-U16 backing dtype override is supported")
        header[name] = {"dtype": stored_dtype, "shape": list(a.shape), "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    _write_parts(path, header, ((a, None, None) for a in arrays.values()))


def _write_parts(path, header, parts):
    path = Path(path)
    raw = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for source, start, size in parts:
            if isinstance(source, np.ndarray):
                source.tofile(out)
            else:
                with Path(source).open("rb") as f:
                    f.seek(start)
                    while size:
                        block = f.read(min(size, 8 << 20))
                        if not block:
                            raise ValueError("Truncated assembly source")
                        out.write(block)
                        size -= len(block)
        out.flush()
        os.fsync(out.fileno())
    from pair1_contract import read_header
    read_header(tmp)
    os.replace(tmp, path)


def fused_engram(gate):
    """Compatible product representation, NOT recovery of original q/k factors."""
    gate = np.asarray(gate)
    if gate.dtype != np.float32 or not np.isfinite(gate).all():
        raise ValueError("Fused Engram gate must be finite F32; no BF16 cast is permitted")
    return gate.copy(), np.ones_like(gate)


def rate_plan(experts, n=28, overrides=None):
    if n not in RATES or experts <= 0:
        raise ValueError("Supported pilot rates: n24/28/32/40")
    plan = {str(e): {p: n for p in PROJECTIONS} for e in range(experts)}
    for e, values in (overrides or {}).items():
        if e not in plan or not isinstance(values, dict):
            raise ValueError("Rate plan has unknown expert")
        for p, rate in values.items():
            if p not in PROJECTIONS or type(rate) is not int or rate not in RATES:
                raise ValueError("Invalid projection rate")
            plan[e][p] = rate
    return plan


def encode_matrix(weight_io, n, *, seed=0, backend="cpu", batch=1, hessian=None):
    """New scales; serialized F16 rounding precedes inversion and LDLQ."""
    import torch
    import stq_quality as quality
    import pair1_contract as contract
    import pair1_cuda
    w = np.asarray(weight_io, dtype=np.float32)
    if w.ndim != 2 or any(d % 128 for d in w.shape) or not np.isfinite(w).all() or n not in RATES:
        raise ValueError("PAIR1 matrix must be finite [in,out] on the H128 grid")
    rng = np.random.default_rng(seed)
    signs = [rng.choice([-1.0, 1.0], d) for d in w.shape]
    _, su, sv = quality.regularize(w, *signs)
    su, sv = su.astype(np.float16), sv.astype(np.float16)
    inner = quality.target_inner(w, su, sv)
    perm = contract.core_perm()

    def callback(tiles):
        x = torch.from_numpy(np.ascontiguousarray(tiles.reshape(-1, 256)[:, perm], dtype=np.float32))
        if backend == "cuda":
            x = x.to("cuda")
        packed, _ = pair1_cuda.encode_tiles(x, n, batch=batch, backend="torch" if backend == "cpu" else backend)
        return packed, contract.decode_tiles(packed).reshape(-1, 16, 16).astype(np.float64)

    if hessian is None:
        i, o = w.shape
        tiles = inner.reshape(i // 16, 16, o // 16, 16).transpose(0, 2, 1, 3)
        packed, _ = callback(tiles)
        packed = packed.reshape(i // 16, o // 16, n)
    elif np.asarray(hessian).ndim == 1:
        blocks = quality.prepare_diagonal_ldl(hessian, su)
        packed, _ = quality.ldlq_blocks(inner, blocks, callback)
    else:
        L, _, _ = quality.prepare_ldl(quality.rotated_hessian(hessian, su))
        packed, _ = quality.ldlq(inner, L, callback)
    reconstructed = contract.reconstruct(packed, su, sv)
    error = reconstructed.astype(np.float64) - w
    weighted = None
    if hessian is not None:
        weighted = (quality.weighted_error_diagonal(w, reconstructed, hessian) if np.asarray(hessian).ndim == 1
                    else quality.weighted_error(w, reconstructed, hessian))
    metrics = {"weights": w.size, "weighted_error": weighted,
               "payload_bits": int(packed.nbytes * 8),
               "scale_bits": int((su.nbytes + sv.nbytes) * 8),
               "sse": float(np.square(error).sum()), "calibrated": hessian is not None}
    return {"trellis": packed, "suh": su, "svh": sv}, metrics


def source_pin(reader, params):
    from orca_reader import REPO, REVISION
    modules = ("orca_export.py", "orca_reader.py", "pair1_reference.py", "pair1_cuda.py",
               "pair1_cuda.cu", "stq_quality.py", "pair1_contract.py", "stq.py", "spec1_w15.txt", "hub.py", "spec1_quality.py", "imatrix_reader.py")
    return {"repo": REPO, "revision": REVISION, "params": params,
            "encoder_sha256": {p: digest(HERE / p) for p in modules if (HERE / p).exists()},
            "catalog": {name: {"path": t.path, "dims": list(t.dims), "quant": t.quant,
                                  "offset": t.offset, "data_start": t.data_start}
                        for name, t in sorted(reader.tensors.items())},
            "dspark": "omitted-donor-absent", "engram": "q=fused-F32,k=ones-F32;not-original-factors",
            "sushi_source": "d5e876e", "status": "pilot-not-final-1.75bpw"}


def projection(reader, manifest, layer, expert, proj, n, params, hessian=None):
    key = f"expert.{expert}.{proj}"
    filename = key + ".safetensors"
    if not manifest.ready(key):
        native = f"layers.{layer}.ffn.experts.{expert}.{PROJECTIONS[proj]}.weight"
        dense = reader.read_native(native, sushi=True)
        started = time.monotonic()
        arrays, metrics = encode_matrix(dense, n, seed=params["seed"], backend=params["backend"],
                                        batch=params["batch"], hessian=hessian)
        metrics["elapsed_seconds"] = time.monotonic() - started
        metrics["relative_frobenius"] = float(np.sqrt(metrics["sse"] / max(float(np.square(dense.astype(np.float64)).sum()), 1e-300)))
        write_safetensors(manifest.root / filename, arrays, {"artifact": "single-projection-pilot", "source": native})
        manifest.finish(key, filename, metrics)
    return manifest.root / filename


def assemble_layer(manifest, layer, groups, dense_files, quant):
    from pair1_contract import read_header, validate_routed, validate_stamp, validate_permutation
    header, parts, cursor = {}, [], 0

    def append(name, sources, stacked=False):
        nonlocal cursor
        infos, ranges = [], []
        for path, source_name in sources:
            h = read_header(path)
            info = h[source_name]
            with Path(path).open("rb") as f:
                data_start = 8 + int.from_bytes(f.read(8), "little")
            start, end = info["data_offsets"]
            infos.append(info)
            ranges.append((path, data_start + start, end - start))
        first = infos[0]
        if any(x["shape"] != first["shape"] or x["dtype"] != first["dtype"] for x in infos):
            raise ValueError("Cannot concatenate incompatible artifacts")
        size = sum(x[2] for x in ranges)
        header[name] = {"dtype": first["dtype"], "shape": ([len(infos)] if stacked else []) + first["shape"],
                        "data_offsets": [cursor, cursor + size]}
        cursor += size
        parts.extend(ranges)

    permutation = [e for _, ids in groups for e in ids]
    validate_permutation(permutation, len(permutation))
    for g, (_, ids) in enumerate(groups):
        for p in PROJECTIONS:
            for leaf in ("trellis", "suh", "svh"):
                append(f"model.layers.{layer}.mlp.switch_mlp.{p}_proj.g{g}.{leaf}",
                       [(manifest.root / f"expert.{e}.{p}.safetensors", leaf) for e in ids], True)
    for path in dense_files:
        h = read_header(path)
        for name in h:
            if name == "__metadata__":
                continue
            # Routing logits and bias must follow the regrouped expert order.
            if name in (f"layers.{layer}.ffn.gate.weight", f"layers.{layer}.ffn.gate.bias",
                        f"layers.{layer}.ffn.gate.bias_vl"):
                from pair1_contract import read_tensors
                a = read_tensors(path)[name][permutation]
                reordered = manifest.root / (path.stem + "." + name + ".permuted.safetensors")
                write_safetensors(reordered, {name: a}, {"artifact": "permuted-router"})
                append(name, [(reordered, name)])
            else:
                append(name, [(path, name)])
    hidden = header[f"model.layers.{layer}.mlp.switch_mlp.gate_proj.g0.suh"]["shape"][-1]
    intermediate = header[f"model.layers.{layer}.mlp.switch_mlp.gate_proj.g0.svh"]["shape"][-1]
    validate_routed(header, quant, hidden, intermediate, len(permutation))
    metadata = {"format": "stq", "codebook": "pair1", "window": "16", "k": str(quant["k"]),
                "pack_revision": hashlib.sha256(json.dumps(manifest.data["pin"], sort_keys=True).encode()).hexdigest(),
                "status": "pilot-dense-F32-trunk-not-final-pack", "engram_representation": "fused-F32-times-one-F32"}
    validate_stamp(metadata, quant)
    header["__metadata__"] = metadata
    _write_parts(manifest.root / f"model-layer-{layer:02d}.safetensors", header, parts)
    manifest.data["expert_permutation"] = permutation
    manifest.data["quantization"] = quant
    manifest.finish("layer", f"model-layer-{layer:02d}.safetensors")


def export(a):
    from orca_reader import OrcaReader
    from pair1_contract import PAIR1_CONFIG
    reader = OrcaReader(a.catalog) if a.catalog else OrcaReader()
    experts = reader.tensors[f"blk.{a.layer}.ffn_gate_exps.weight"].shape[0]
    overrides = json.loads(Path(a.rates).read_text()) if a.rates else None
    rates = rate_plan(experts, a.n, overrides)
    importance = None
    if getattr(a, "imatrix", False):
        if a.calibration:
            raise ValueError("Choose --imatrix or --calibration, not both")
        from imatrix_reader import ImatrixReader
        importance = ImatrixReader()
    trunk_mode = getattr(a, "trunk", "dense")
    trunk_h_path = getattr(a, "trunk_calibration", None)
    if trunk_mode == "spec1" and not trunk_h_path:
        raise ValueError("SPEC1 trunk requires explicit per-tensor calibration Hessians")
    params = {"layer": a.layer, "expert": a.expert, "projection": a.projection, "rates": rates,
              "backend": a.backend, "seed": a.seed, "batch": a.batch, "trunk": trunk_mode,
              "trunk_n": getattr(a, "trunk_n", 96),
              "imatrix": importance.identity() if importance is not None else None,
              "importance_normalization": "layer" if importance is not None else None,
              "trunk_calibration": {"sha256": digest(trunk_h_path)} if trunk_h_path else None,
              "calibration": {"path": str(Path(a.calibration).resolve()), "sha256": digest(a.calibration)} if a.calibration else None,
              "calibration_provenance": {"sha256": digest(a.calibration_provenance)} if getattr(a, "calibration_provenance", None) else None}
    manifest = Manifest(a.output, source_pin(reader, params))
    manifest.data["scope_status"] = "layer-artifact-not-complete-pack"
    if trunk_mode == "spec1":
        manifest.data["trunk_policy"] = "Only exact native keys supplied in trunk Hessian NPZ become SPEC1; remaining weights stay explicit F32"
        manifest.data["trunk_quant"] = dict(format="stq", codebook="spec1", window=15, k=params["trunk_n"] / 16,
                                             codebook_table="spec1-w15", codebook_digest="37f72431")
    calibration = np.load(a.calibration, allow_pickle=False) if a.calibration else None
    ids = range(experts) if a.expert is None else [a.expert]
    if any(e < 0 or e >= experts for e in ids):
        raise ValueError("Expert index outside donor")
    for e in ids:
        for p in ([a.projection] if a.projection else PROJECTIONS):
            native = f"layers.{a.layer}.ffn.experts.{e}.{PROJECTIONS[p]}.weight"
            record = importance.read_native(native) if importance is not None else None
            h = record.weights("layer") if record is not None else calibration[native] if calibration is not None else None
            projection(reader, manifest, a.layer, e, p, rates[str(e)][p], params, h)
            if record is not None:
                manifest.data.setdefault("importance_provenance", {})[native] = record.provenance
                atomic_json(manifest.path, manifest.data)
    if a.expert is not None or a.projection is not None:
        return manifest
    if manifest.ready("layer"):
        return manifest
    trunk_calibration = np.load(trunk_h_path, allow_pickle=False) if trunk_h_path else None
    dense_files = []
    for native, donors in sorted(reader.native_sources().items()):
        if not native.startswith(f"layers.{a.layer}.") or ".experts." in native:
            continue
        if native.endswith("engram.embed.weight"):
            manifest.data["external_requirements"] = [f"layers.{a.layer}.engram.embed.weight", f"layers.{a.layer}.engram.embed.scale"]
            continue
        is_spec1 = trunk_mode == "spec1" and trunk_calibration is not None and native in trunk_calibration
        key = ("spec1." if is_spec1 else "dense.") + native
        filename = key + ".safetensors"
        if not manifest.ready(key):
            if native.endswith("engram.k_weight"):
                continue  # q work item stores both compatible factors.
            if native.endswith("engram.q_weight"):
                gate = reader.read(f"blk.{a.layer}.engram_gate.weight")
                q, k = fused_engram(gate)
                arrays = {native: q, native.replace("q_weight", "k_weight"): k}
            else:
                arrays = {native: np.asarray(reader.read_native(native), dtype=np.float32)}
            if is_spec1:
                from spec1_quality import encode_trunk
                from pair1_contract import validate_trunk, read_header
                if not native.endswith(".weight") or len(arrays) != 1:
                    raise ValueError("SPEC1 calibration key must identify one linear weight")
                weight = arrays[native]
                h = trunk_calibration[native]
                if native.endswith("attn.wo_a.weight"):
                    if h.ndim not in (2, 3) or weight.shape[0] % h.shape[0]:
                        raise ValueError("Grouped wo_a needs explicit group Hessians")
                    weight = weight.reshape(h.shape[0], -1, weight.shape[1]).transpose(0, 2, 1)
                else:
                    weight = weight.T
                encoded = encode_trunk(weight, h, n=params["trunk_n"], seed=a.seed, name=native,
                                       device="cuda" if a.backend == "cuda" else "cpu", batch=a.batch)
                base = native[:-7]
                arrays = {base + "." + leaf: encoded[leaf] for leaf in ("trellis", "suh", "svh")}
                meta = {"format": "stq", "codebook": "spec1", "window": "15", "k": str(params["trunk_n"] / 16)}
                write_safetensors(manifest.root / filename, arrays, meta)
                tq = dict(format="stq", codebook="spec1", window=15, k=params["trunk_n"] / 16,
                          codebook_table="spec1-w15", codebook_digest="37f72431")
                validate_trunk(read_header(manifest.root / filename), tq)
                manifest.finish(key, filename, {"calibration_loss": encoded["calibration_loss"], "n": params["trunk_n"]})
                manifest.data["trunk_quant"] = tq
            else:
                write_safetensors(manifest.root / filename, arrays, {"artifact": "donor-dense-F32-pilot"})
                manifest.finish(key, filename)
        if not is_spec1:
            dense_files.append(manifest.root / filename)
    grouped = {}
    for e in range(experts):
        signature = tuple(rates[str(e)][p] for p in PROJECTIONS)
        grouped.setdefault(signature, []).append(e)
    if len(grouped) > 32:
        raise ValueError("Consumer supports at most 32 routed groups")
    quant = dict(PAIR1_CONFIG, k=max(max(t) for t in grouped) / 16)
    assemble_layer(manifest, a.layer, sorted(grouped.items()), dense_files, quant)
    return manifest


def conversion_plan(reader):
    """Enumerate every source tensor, including unsupported transformations."""
    from orca_reader import native_names
    entries, expected = [], set()
    for name, tensor in sorted(reader.tensors.items()):
        try:
            destinations = list(native_names(name, tensor))
            expected.update(destinations)
            if name == "per_layer_token_embd.weight":
                transform = "engram_export:bounded-IQ4_NL-to-FP8-lossy"
                status = "external-streaming-adapter"
            elif name.endswith("engram_gate.weight"):
                transform, status = "F32-product-times-one-not-original-factors", "supported-derived"
            elif "_exps.weight" in name:
                transform, status = "all-experts-new-PAIR1-with-router-permutation", "supported-pilot"
            elif name.startswith("v.") or name.startswith("mm."):
                transform, status = "reader-native-reshape-or-concat-F32", "globals-writer"
            else:
                transform, status = "reader-native-F32-or-calibrated-SPEC1", "supported-pilot"
            entries.append({"donor": name, "shape": list(tensor.shape), "quant": tensor.quant,
                            "source_bytes": tensor.nbytes, "destinations": destinations,
                            "transform": transform, "status": status})
        except (KeyError, ValueError) as error:
            entries.append({"donor": name, "status": "unsupported", "error": str(error), "destinations": []})
    estimates = {}
    for entry in entries:
        if entry["status"] == "unsupported":
            continue
        shape = entry["shape"]
        count = int(np.prod(shape, dtype=np.int64))
        name = entry["donor"]
        if name == "per_layer_token_embd.weight":
            category, size = "native-FP8-engram", count + count // 32
        elif "_exps.weight" in name:
            category, size = "routed-n28-including-F16-scales", count * 28 // 128 + 2 * shape[0] * (shape[1] + shape[2])
        elif name in ("output.weight", "token_embd.weight"):
            category, size = "F32-embedding-head", count * 4
        elif name.startswith(("v.", "mm.")):
            category, size = "BF16-vision", count * 2
        else:
            category, size = "F32-other-trunk", count * 4
        estimates[category] = estimates.get(category, 0) + size
    return {"status": "integration-scaffold-not-complete-conversion", "donor_tensors": len(entries),
            "size_estimate": {"bytes_by_category": estimates, "total_bytes": sum(estimates.values()),
                              "excludes": "headers;manifests;temporary-duplicate-work-items;calibration;extra-fused-gate-factor",
                              "not_final_budget": True}, 
            "native_destinations": len(expected), "mappings": entries,
            "omissions": [{"component": "DSpark", "reason": "absent from donor; never grafted"}],
            "required_assets": ["config.json", "tokenizer.json", "tokenizer_config.json", "engram_token_map.json"],
            "blockers": ["Donor GGUF metadata -> native config/tokenizer equivalence requires explicit verified assets",
                         "Final SPEC1 trunk calibration and whole-pack byte allocation are not quality-validated",
                         "Full model loader/KLD/behavioral validation not performed"]}


def gguf_metadata(raw):
    """Read bounded GGUF metadata including arrays needed by native assets."""
    import io
    f = io.BytesIO(raw)
    def read(fmt):
        size = struct.calcsize("<" + fmt)
        b = f.read(size)
        if len(b) != size:
            raise ValueError("Truncated GGUF metadata")
        return struct.unpack("<" + fmt, b)[0]
    def string():
        size = read("Q")
        if size > len(raw):
            raise ValueError("Invalid GGUF string")
        b = f.read(size)
        if len(b) != size:
            raise ValueError("Truncated GGUF string")
        return b.decode("utf-8")
    fm = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            subtype, count = read("I"), read("Q")
            if count > len(raw):
                raise ValueError("Invalid GGUF array length")
            return [value(subtype) for _ in range(count)]
        return read(fm[kind])
    if f.read(4) != b"GGUF" or read("I") != 3:
        raise ValueError("Expected GGUFv3")
    read("Q")
    count = read("Q")
    metadata = {}
    for _ in range(count):
        key = string()
        metadata[key] = value(read("I"))
    return metadata


def prepare_assets(directory):
    """Materialize only checksum-proven structural assets and verbatim donor KV."""
    from hub import rng
    from orca_reader import REPO, REVISION, shard_path
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    revision = "08c8657f02291e11b6f6b63e0e16014f8e748518"
    records = {}
    for name, size, sha in (
        ("config.json",3311,"8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"),
        ("tokenizer.json",6367257,"c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"),
        ("tokenizer_config.json",801,"6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547")):
        data = rng(base,name,0,size-1,revision)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Pinned native asset checksum mismatch: " + name)
        target = root / name
        if target.exists() and digest(target) != sha:
            raise ValueError("Refusing to overwrite different asset: " + name)
        target.write_bytes(data)
        records[name] = {"source_repo":base,"source_revision":revision,"sha256":sha,
                         "proof":"DONOR_ASSETS.md:exact-geometry-or-token-ID-and-merge-proof"}
    kv = gguf_metadata(rng(REPO,shard_path(1),0,5781032,REVISION))
    template = kv["tokenizer.chat_template"].encode("utf-8")
    if hashlib.sha256(template).hexdigest() != "84e42b896953316f4a4713d961219ae9c7ea77effa90fdea189ede586f765ebb":
        raise ValueError("Donor chat template checksum mismatch")
    (root / "donor_chat_template.jinja").write_bytes(template)
    maps = [(key, value) for key,value in kv.items() if "token_map" in key and isinstance(value,list)]
    if len(maps) != 1:
        raise ValueError("Expected one exact embedded donor token map")
    key, token_map = maps[0]
    map_bytes = np.asarray(token_map,dtype='<u4').tobytes()
    if hashlib.sha256(map_bytes).hexdigest() != "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481":
        raise ValueError("Donor Engram map checksum mismatch")
    atomic_json(root / "engram_token_map.json",token_map)
    hashing = {key:value for key,value in kv.items() if ".ple." in key or ".engram." in key}
    atomic_json(root / "donor_engram_metadata.json",hashing)
    records["donor_chat_template.jinja"] = {"source_repo":REPO,"source_revision":REVISION,
                                               "sha256":hashlib.sha256(template).hexdigest(),"mode":"verbatim-not-base-template"}
    records["engram_token_map.json"] = {"source_repo":REPO,"source_revision":REVISION,
                                           "embedded_key":key,"u32le_sha256":hashlib.sha256(map_bytes).hexdigest()}
    atomic_json(root / "asset_provenance.json",records)
    atomic_json(root / "assets.json",{name:str(root/name) for name in ("config.json","tokenizer.json","tokenizer_config.json","engram_token_map.json","donor_chat_template.jinja","donor_engram_metadata.json","asset_provenance.json")})
    return records


def bf16_bits(values):
    """Round finite F32 to BF16 nearest-even; original BF16 donor is exact."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("BF16 donor must be finite")
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def write_globals(reader, directory):
    """One native global/vision item per shard, donor only; no full model assembly."""
    params = {"mode": "globals-text-F32-vision-BF16", "vision": True}
    manifest = Manifest(directory, source_pin(reader, params))
    for native in sorted(reader.native_sources()):
        if native.startswith("layers."):
            continue
        key = "global." + native
        filename = key + ".safetensors"
        if not manifest.ready(key):
            array = np.asarray(reader.read_native(native), dtype=np.float32)
            vision = native.startswith(("vision.", "aligner.")) or native in ("image_start", "image_end", "image_newline")
            stored = bf16_bits(array) if vision else array
            write_safetensors(manifest.root / filename, {native: stored},
                              {"artifact": "donor-vision-BF16" if vision else "donor-global-F32-scaffold"},
                              {native: "BF16"} if vision else None)
            manifest.finish(key, filename)
    return manifest


def index_shards(plan, shard_paths, output, assets=None):
    """Explicit partial index and name coverage. Never marks the model loadable."""
    from pair1_contract import read_header
    mapping, total, coverage = {}, 0, set()
    manifests = []
    for raw in shard_paths:
        path = Path(raw).resolve()
        header = read_header(path)
        if header.get("__metadata__", {}).get("status", "").startswith("sample"):
            raise ValueError("Sample Engram shard cannot enter full-pack index")
        for layer, rows in ((1, 384006168), (14, 384016682)):
            prefix = f"layers.{layer}.engram.embed."
            if prefix + "weight" in header:
                w, s = header[prefix + "weight"], header.get(prefix + "scale")
                if w["dtype"] != "F8_E4M3" or w["shape"] != [rows, 256] or s is None or s["dtype"] != "F8_E8M0" or s["shape"] != [rows, 8]:
                    raise ValueError("Native Engram weight/scale must be complete and co-located")
        total += path.stat().st_size
        manifests.append({"path": str(path), "bytes": path.stat().st_size})
        for name in header:
            if name == "__metadata__":
                continue
            if name in mapping:
                raise ValueError(f"Duplicate destination tensor: {name}")
            mapping[name] = str(path)
            coverage.add(name)
            if name.endswith(".trellis"):
                coverage.add(name[:-8] + ".weight")
    expected = {n for m in plan["mappings"] for n in m["destinations"]}
    # Routed grouped names replace individual expert names. Require a complete
    # independently validated layer AND its explicit source-ID permutation.
    from pair1_contract import validate_routed, validate_permutation
    for path_record in manifests:
        path = Path(path_record["path"])
        header = read_header(path)
        layers = {int(k.split(".")[2]) for k in header if k.startswith("model.layers.") and ".switch_mlp." in k}
        for layer in layers:
            manifest_path = path.parent / "manifest.json"
            state = json.loads(manifest_path.read_text())
            permutation = state["expert_permutation"]
            validate_permutation(permutation, len(permutation))
            quant = state["quantization"]
            prefix = f"model.layers.{layer}.mlp.switch_mlp.gate_proj.g0."
            hidden, inter = header[prefix + "suh"]["shape"][-1], header[prefix + "svh"]["shape"][-1]
            validate_routed(header, quant, hidden, inter, len(permutation))
            if state["pin"]["params"]["layer"] != layer:
                raise ValueError("Layer manifest mismatch")
            for expert in permutation:
                for projection in PROJECTIONS.values():
                    coverage.add(f"layers.{layer}.ffn.experts.{expert}.{projection}.weight")
    asset_records = {}
    for name, path_value in (assets or {}).items():
        path = Path(path_value).resolve()
        allowed = set(plan["required_assets"]) | {"donor_chat_template.jinja", "donor_engram_metadata.json", "asset_provenance.json"}
        if name not in allowed or path.name != name:
            raise ValueError("Unexpected asset; exact native basename required")
        if path.suffix == ".json":
            json.loads(path.read_text())
        asset_records[name] = {"path": str(path), "sha256": digest(path), "equivalence": "user-supplied-not-verified"}
    missing = sorted(expected - coverage)
    result = {"metadata": {"total_size": total, "status": "scaffold-not-loader-approved"}, "weight_map": mapping,
              "coverage": {"missing": missing, "unexpected": sorted(coverage - expected),
                           "unsupported_sources": [m["donor"] for m in plan["mappings"] if m["status"] == "unsupported"]},
              "assets": asset_records, "missing_assets": sorted(set(plan["required_assets"]) - set(asset_records)),
              "omissions": plan["omissions"], "blockers": plan["blockers"], "shards": manifests}
    atomic_json(output, result)
    return result


def validate_discovery(root, weight_map):
    """Mirror Sushi root model*.safetensors discovery, not index-only coverage."""
    from pair1_contract import read_header
    root = Path(root).resolve()
    discovered = {}
    for path in sorted(root.glob("model*.safetensors")):
        if path.resolve().parent != root:
            raise ValueError("Discovered shard escapes pack root")
        for name in read_header(path):
            if name == "__metadata__":
                continue
            if name in discovered:
                raise ValueError("Duplicate tensor in loader-discovered shards")
            discovered[name] = path.name
    for name, filename in weight_map.items():
        path = root / filename
        if Path(filename).name != filename or path.resolve().parent != root or not filename.startswith("model") or not filename.endswith(".safetensors"):
            raise ValueError("Index shard must be a root model*.safetensors basename")
    if discovered != weight_map:
        raise ValueError("Loader-discovered tensors differ from index ownership")
    return discovered


def materialize_pack(plan, shard_paths, directory, assets=None):
    """Copy immutable work artifacts to loader-discoverable root filenames.

    A materialized partial pack is never declared inference/quality ready.
    Sources and work-item manifests are preserved; no cleanup or hard links.
    """
    import shutil
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = index_shards(plan, shard_paths, root / "coverage.json", assets)
    from pair1_contract import read_header, PAIR1_CONFIG
    source_headers = [(Path(record["path"]), read_header(record["path"])) for record in report["shards"]]
    expert_ns, trunk_ns = [], []
    for source, header in source_headers:
        for key, info in header.items():
            if key.endswith(".trellis"):
                (expert_ns if ".switch_mlp." in key else trunk_ns).append(info["shape"][-1])
    policy = {"expert_quant": dict(PAIR1_CONFIG, k=max(expert_ns) / 16) if expert_ns else None,
              "trunk_quant": dict(format="stq", codebook="spec1", window=15, k=max(trunk_ns) / 16,
                                  codebook_table="spec1-w15", codebook_digest="37f72431") if trunk_ns else None}
    revision = hashlib.sha256(json.dumps({"policy": policy, "source_headers": source_headers and [
        {"header": header, "bytes": source.stat().st_size, "sha256": digest(source)} for source, header in source_headers],
        "assets": report["assets"]}, sort_keys=True).encode()).hexdigest()
    names = {}
    for number, record in enumerate(report["shards"]):
        source = Path(record["path"])
        from pair1_contract import read_header
        header = read_header(source)
        keys = [k for k in header if k != "__metadata__"]
        kind = ("engram" if any(".engram.embed." in k for k in keys) else
                "experts" if any(".switch_mlp." in k for k in keys) else
                "trunk" if any(k.startswith("layers.") for k in keys) else "global")
        target = root / f"model-{kind}-{number:05d}.safetensors"
        metadata = dict(header.get("__metadata__", {}), pack_revision=revision,
                        source_sha256=digest(source), stage="materialized-not-approved")
        if kind == "experts":
            metadata.update(format="stq", codebook="pair1", window="16", k=str(policy["expert_quant"]["k"]))
        elif any(k.endswith(".trellis") for k in keys):
            metadata.update(format="stq", codebook="spec1", window="15", k=str(policy["trunk_quant"]["k"]))
        header["__metadata__"] = metadata
        if target.exists():
            if read_header(target) != header:
                raise ValueError("Existing materialized shard differs from expected header")
            with target.open("rb") as f:
                data_start = 8 + int.from_bytes(f.read(8), "little")
                from engram_export import range_digest
                payload_size = target.stat().st_size - data_start
                actual = range_digest(f, data_start, payload_size)
            with source.open("rb") as f:
                data_start = 8 + int.from_bytes(f.read(8), "little")
                expected_payload = range_digest(f, data_start, source.stat().st_size - data_start)
            if actual != expected_payload:
                raise ValueError("Existing materialized shard payload changed")
        else:
            with source.open("rb") as f:
                data_start = 8 + int.from_bytes(f.read(8), "little")
            _write_parts(target, header, [(source, data_start, source.stat().st_size - data_start)])
        names[str(source)] = target.name
    weight_map = {key: names[path] for key, path in report["weight_map"].items()}
    validate_discovery(root, weight_map)
    # Supplied assets remain explicitly unverified, even after safe materialization.
    for name, record in report["assets"].items():
        target = root / name
        source = Path(record["path"])
        if target.exists() and digest(target) != record["sha256"]:
            existing_config = json.loads(target.read_text()) if name == "config.json" else {}
            if name != "config.json" or existing_config.get("pack_revision", {}).get("id") != revision:
                raise ValueError("Existing materialized asset differs from supplied asset")
        if not target.exists():
            shutil.copyfile(source, target)
    if "config.json" in report["assets"]:
        config = json.loads(Path(report["assets"]["config.json"]["path"]).read_text())
        for key, value in policy.items():
            config.pop(key, None)
            if value is not None:
                config[key] = value
        config["pack_revision"] = {"id": revision, "unstamped": []}
        for section in (config, config.get("text_config", {})):
            if isinstance(section, dict):
                section["num_nextn_predict_layers"] = 0
                for key in list(section):
                    if key.startswith("dspark_"):
                        del section[key]
        atomic_json(root / "config.json", config)
    total_size = sum((root / filename).stat().st_size for filename in set(weight_map.values()))
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    atomic_json(root / "model.safetensors.index.json", index)
    report["weight_map"] = weight_map
    report["discovery"] = {"exact_index_ownership": True, "tensor_count": len(weight_map)}
    report["pack_revision"] = revision
    report["quantization_policy"] = policy
    report["metadata"]["status"] = "materialized-not-loader-or-quality-approved"
    report["completion_gates"] = {"native_names_complete": not report["coverage"]["missing"] and not report["coverage"]["unsupported_sources"],
                                  "assets_present": not report["missing_assets"],
                                  "discovery_exact": True,
                                  "asset_proof_validated": False,
                                  "whole_pack_budget_validated": False,
                                  "sushi_load_selftest_passed": False,
                                  "heldout_quality_passed": False}
    report["ready"] = all(report["completion_gates"].values())
    atomic_json(root / "coverage.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--plan", action="store_true", help="Enumerate text+vision mapping; headers only")
    parser.add_argument("--prepare-assets", action="store_true", help="Materialize pinned verified native assets and verbatim donor template/map")
    parser.add_argument("--globals", action="store_true", help="Write donor global/vision F32 scaffold shards")
    parser.add_argument("--index-shards", nargs="+", help="Write explicit partial index/coverage from emitted shards")
    parser.add_argument("--assets", help="JSON exact native asset basename -> supplied local file")
    parser.add_argument("--materialize", action="store_true", help="With --index-shards: copy to root model*.safetensors and relative index")
    parser.add_argument("--trunk", choices=("dense", "spec1"), default="dense")
    parser.add_argument("--trunk-n", type=int, default=96)
    parser.add_argument("--trunk-calibration", help="NPZ native trunk weight names -> input Hessians")
    parser.add_argument("--expert", type=int)
    parser.add_argument("--projection", choices=PROJECTIONS)
    parser.add_argument("--n", type=int, choices=RATES, default=28)
    parser.add_argument("--rates", help='JSON {"expert_id":{"gate":24,"up":32,"down":40}}')
    parser.add_argument("--calibration", help="NPZ native expert weight names -> input Hessians")
    parser.add_argument("--imatrix", action="store_true", help="Lazy pinned native donor importance, common-layer normalization; routed experts only")
    parser.add_argument("--calibration-provenance", help="JSON sidecar pinned by SHA256")
    parser.add_argument("--catalog")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    a = parser.parse_args()
    if a.batch < 1 or (a.layer is not None and a.layer < 0):
        parser.error("layer must be nonnegative and batch positive")
    if a.prepare_assets:
        print(json.dumps(prepare_assets(a.output), indent=2))
        return
    if a.plan or a.globals or a.index_shards:
        from orca_reader import OrcaReader
        reader = OrcaReader(a.catalog) if a.catalog else OrcaReader()
        reader.add_vision()
        plan = conversion_plan(reader)
        if a.plan:
            atomic_json(a.output, plan)
        elif a.globals:
            print(write_globals(reader, a.output).path)
        else:
            assets = json.loads(Path(a.assets).read_text()) if a.assets else None
            result = (materialize_pack(plan, a.index_shards, a.output, assets) if a.materialize
                      else index_shards(plan, a.index_shards, a.output, assets))
            print(json.dumps({"missing_tensors": len(result["coverage"]["missing"]), "status": result["metadata"]["status"]}))
        return
    if a.layer is None:
        parser.error("--layer is required for expert/layer export")
    print(export(a).path)


if __name__ == "__main__":
    main()
