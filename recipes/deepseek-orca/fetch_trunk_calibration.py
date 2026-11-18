"""Fetch only pinned native trunk input diagonals; never weights or dense Hessians.

Run with the work venv. NPZ keys are exact native names, wo_a values are [8,4096].
Sidecar records raw sums/counts/source byte hashes and explicit missing entries.
"""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from imatrix_reader import ImatrixReader
from orca_reader import OrcaReader


def fetch_trunk(reader, imatrix, cache_dir=None):
    arrays, records, skipped = {}, {}, {}
    source = imatrix.identity()
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = cache_dir / "source.json"
        if identity.exists() and json.loads(identity.read_text()) != source:
            raise ValueError("Calibration cache source identity mismatch")
        if not identity.exists():
            identity.write_text(json.dumps(source, sort_keys=True) + "\n")

    def one(item):
        native, donors = item
        if not native.startswith("layers.") or ".experts." in native or not native.endswith(".weight"):
            return
        tensor = reader.tensors[donors[0]]
        if native.endswith("engram.embed.weight") or len(tensor.shape) != 2:
            return
        if native.endswith("ffn.gate.weight"):
            skipped[native] = {"reason": "router_requires_F32"}
            return
        if any(d % 128 for d in tensor.shape):
            skipped[native] = {"reason": "unsupported_SPEC1_H128_grid", "shape": list(tensor.shape)}
            return
        cached_array = cache_dir / (native + ".npy") if cache_dir is not None else None
        cached_meta = cache_dir / (native + ".json") if cache_dir is not None else None
        if cached_array is not None and cached_array.exists() and cached_meta.exists():
            meta = json.loads(cached_meta.read_text())
            if hashlib.sha256(cached_array.read_bytes()).hexdigest() != meta["cache_sha256"]:
                raise ValueError(f"Corrupt calibration cache: {native}")
            diagonal = np.load(cached_array, allow_pickle=False)
            if list(diagonal.shape) != meta["shape"] or not np.isfinite(diagonal).all() or np.any(diagonal < 0):
                raise ValueError(f"Invalid cached diagonal: {native}")
            arrays[native], records[native] = diagonal, meta
            print(f"RESUMED {native}", flush=True)
            return
        try:
            if native.endswith("attn.wo_a.weight"): 
                importance = [imatrix.read_native(native, group=g) for g in range(8)]
                diagonal = np.stack([r.weights("layer") for r in importance])
                provenance = [r.provenance for r in importance]
                if diagonal.shape != (8, tensor.shape[-1]):
                    raise ValueError("Grouped diagonal does not match donor native input width")
            else:
                record = imatrix.read_native(native)
                diagonal = record.weights("layer")
                provenance = [record.provenance]
                if diagonal.shape != (tensor.shape[-1],):
                    raise ValueError("Diagonal does not match donor native input width")
        except KeyError as e:
            skipped[native] = {"reason": "missing_imatrix_entry", "detail": str(e)}
            return
        except ValueError as e:
            # Invalid observed data must not quietly turn into an uncalibrated trunk.
            raise ValueError(f"{native}: {e}") from e
        arrays[native] = diagonal
        records[native] = {"shape": list(diagonal.shape), "normalization": "raw_sum2/common_layer_count",
                           "layer_count": provenance[0]["layer_count"], "sources": provenance}
        if cached_array is not None:
            with cached_array.open("wb") as f:
                np.save(f, diagonal, allow_pickle=False)
            records[native]["cache_sha256"] = hashlib.sha256(cached_array.read_bytes()).hexdigest()
            cached_meta.write_text(json.dumps(records[native], sort_keys=True) + "\n")
        print(f"CALIBRATED {native} {list(diagonal.shape)}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, sorted(reader.native_sources().items())))
    if not arrays:
        raise ValueError("No calibrated trunk matrices")
    return arrays, {"source": source, "tensors": records, "skipped": skipped,
                    "coordinates": "native_input", "dense_covariance": False,
                    "missing_policy": "explicit skip; no synthetic importance"}


def self_check():
    class T:
        shape = (1024, 128)
    class Reader:
        tensors = {"donor": T()}
        def native_sources(self):
            return {"layers.0.attn.wo_a.weight": ["donor"],
                    "layers.0.attn.wkv.weight": ["donor"],
                    "layers.0.ffn.gate.weight": ["donor"],
                    "layers.0.attn.indexer.wk.weight": ["donor"]}
    class Record:
        provenance = {"layer_count": 10}
        def weights(self, normalization):
            assert normalization == "layer"
            return np.ones(128)
    class Imatrix:
        def identity(self):
            return {"revision": "test"}
        def read_native(self, name, group=None):
            if name.endswith("wk.weight"):
                raise KeyError("absent")
            return Record()
    arrays, manifest = fetch_trunk(Reader(), Imatrix())
    assert arrays["layers.0.attn.wo_a.weight"].shape == (8, 128)
    assert arrays["layers.0.attn.wkv.weight"].shape == (128,)
    assert manifest["skipped"]["layers.0.ffn.gate.weight"]["reason"] == "router_requires_F32"
    assert manifest["skipped"]["layers.0.attn.indexer.wk.weight"]["reason"] == "missing_imatrix_entry"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="orca-trunk-diagonal.npz")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
        return
    output = Path(a.output).resolve()
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Calibration output already exists; choose a new output path")
    arrays, metadata = fetch_trunk(OrcaReader(), ImatrixReader(), output.with_suffix(".cache"))
    with output.open("xb") as f:
        np.savez(f, **arrays)
    metadata["npz_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata["npz_bytes"] = output.stat().st_size
    with sidecar.open("x") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"output": str(output), "sidecar": str(sidecar),
                      "calibrated": len(arrays), "skipped": len(metadata["skipped"]),
                      "bytes": metadata["npz_bytes"], "sha256": metadata["npz_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
