"""Header-bounded final Orca candidate preflight; never asserts runtime/quality readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from pair1_contract import (ROUTED, read_header, validate_permutation, validate_quant,
                            validate_routed, validate_stamp, validate_trunk)

DEFAULT_COUNTS = dict(spec1=337, dense_trunk=464, globals=269)
ENGRAM_ROWS = {1: 384006168, 14: 384016682}
GEOMETRY = dict(layers=40, hidden=5120, intermediate=2304, experts=384, top_k=6, vocab=129280)
REQUIRED_ASSETS = {"config.json", "tokenizer.json", "tokenizer_config.json", "engram_token_map.json"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path, payload=False):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        if payload:
            n = int.from_bytes(f.read(8), "little")
            f.seek(8 + n)
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_vision_dtype(name, info, source_dtypes=None):
    """Matrices require BF16; vectors preserve their catalog's original precision."""
    if len(info["shape"]) >= 2:
        expected = "BF16"
    else:
        require(source_dtypes is not None and source_dtypes and len(set(source_dtypes)) == 1 and
                source_dtypes[0] in ("F32", "BF16"), "vision vector source dtype missing or mixed")
        expected = source_dtypes[0]
    require(info["dtype"] == expected, "vision dtype differs from matrix-BF16/original-vector policy")


def source_dtypes(native, state):
    catalog = state["pin"]["catalog"]
    global_names = {"image_start": "v.token_embd.img_start", "image_end": "v.token_embd.img_end",
                    "image_newline": "v.image_newline", "vision.norm.weight": "v.post_ln.weight",
                    "vision.patch_embed.proj.bias": "v.patch_embd.bias"}
    donor = global_names.get(native)
    if native.startswith("aligner."):
        donor = native.replace("aligner.w1.", "mm.1.").replace("aligner.w2.", "mm.2.")
    m = re.fullmatch(r"vision\.blocks\.(\d+)\.(norm1|norm2|attn\.wqkv|attn\.wo)\.(weight|bias)", native)
    if m:
        tail = {"norm1": "ln1", "norm2": "ln2", "attn.wqkv": "attn_qkv", "attn.wo": "attn_out"}[m[2]]
        donor = f"v.blk.{m[1]}.{tail}.{m[3]}"
    require(donor in catalog, "native vision vector absent from source catalog")
    return [catalog[donor]["quant"]]


def committed_sources(manifest_paths):
    """Reuse committed hashes, not a fresh scan of hundreds of GB of weight payloads."""
    records, layer_manifests = {}, {}
    for path in manifest_paths:
        path = Path(path).resolve()
        state = load(path)
        if "expert_permutation" in state:
            layer = state["pin"]["params"]["layer"]
            require(layer not in layer_manifests, "duplicate layer manifest")
            layer_manifests[layer] = state
        if "complete" in state:
            for record in state["complete"].values():
                if "file" not in record:
                    continue
                source = (path.parent / record["file"]).resolve()
                require(source.parent == path.parent, "committed artifact escapes manifest directory")
                require(re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")) is not None,
                        "missing committed SHA256")
                require(str(source) not in records, "duplicate committed source")
                records[str(source)] = dict(kind="artifact", sha256=record["sha256"], state=state)
        elif state.get("done") and "chunks" in state:
            require(path.name.endswith(".resume.json"), "Engram resume filename required")
            source = str(path)[:-len(".resume.json")]
            cursor = 0
            for chunk in state["chunks"]:
                require(chunk["row"] == cursor and chunk["rows"] > 0, "Engram committed chunk gaps")
                require(all(re.fullmatch(r"[0-9a-f]{64}", chunk.get(k, "")) for k in ("weight", "scale")),
                        "Engram chunk digest missing")
                cursor += chunk["rows"]
            require(cursor == state["pin"]["rows"], "Engram chunks incomplete")
            require(source not in records, "duplicate Engram source")
            records[source] = dict(kind="engram", state=state)
    return records, layer_manifests


def check_candidate(root, report_path, manifest_paths, *, expected_counts=None, verify_payload=False):
    root = Path(root).resolve()
    report = load(report_path)
    require(not report["coverage"]["missing"] and not report["coverage"]["unsupported_sources"], "native coverage incomplete")
    require(not report["missing_assets"], "required assets missing")
    require(REQUIRED_ASSETS <= set(report["assets"]), "required asset records missing")
    counts = dict(DEFAULT_COUNTS if expected_counts is None else expected_counts)
    require(set(counts) == set(DEFAULT_COUNTS) and all(type(n) is int and n >= 0 for n in counts.values()), "invalid expected counts")
    config = load(root / "config.json")
    text = config.get("text_config", config)
    g = GEOMETRY
    require(config.get("model_type") == "deepseek_v41", "wrong architecture")
    require((text.get("num_hidden_layers"), text.get("hidden_size"), text.get("moe_intermediate_size"),
             text.get("n_routed_experts"), text.get("num_experts_per_tok"), text.get("vocab_size")) ==
            tuple(g[k] for k in ("layers", "hidden", "intermediate", "experts", "top_k", "vocab")), "native geometry mismatch")
    for section in (config, text):
        require(section.get("num_nextn_predict_layers", 0) == 0 and
                not any(k.startswith("dspark_") for k in section), "DSpark donor absent; config must disable it")
    revision = config.get("pack_revision")
    require(isinstance(revision, dict) and isinstance(revision.get("id"), str) and revision["id"] and
            revision.get("unstamped", []) == [], "shared generated revision object required")
    revision = revision["id"]
    require(report.get("pack_revision") == revision, "report revision mismatch")
    eq, tq = config["expert_quant"], config["trunk_quant"]
    en, tn = validate_quant(eq, "pair1"), validate_quant(tq, "spec1")
    require(report.get("quantization_policy") == dict(expert_quant=eq, trunk_quant=tq), "report quant policy mismatch")
    records, layer_manifests = committed_sources(manifest_paths)
    mapping = load(root / "model.safetensors.index.json")["weight_map"]
    require(mapping == report["weight_map"], "report/index mismatch")
    require(all(Path(v).name == v and v.startswith("model") and v.endswith(".safetensors") for v in mapping.values()),
            "index requires root model*.safetensors basenames")
    all_tensors, owners, headers = {}, {}, {}
    source_by_hash = {}
    for item in report["shards"]:
        source = str(Path(item["path"]).resolve())
        require(source in records, "source lacks supplied committed manifest: " + source)
        # Read source HEADER only; the committed digest is a trust boundary, not reverified by default.
        sh = read_header(source)
        record = records[source]
        if record["kind"] == "artifact":
            source_by_hash.setdefault(record["sha256"], []).append((source, sh, record))
        else:
            # Engram manifests commit chunk hashes, not a whole-file digest. Bind by exact header entries.
            source_by_hash.setdefault("engram", []).append((source, sh, record))
    used_sources, payload_checks, materialized_sources = set(), 0, {}
    for path in sorted(root.glob("model*.safetensors")):
        require(path.resolve().parent == root, "shard escapes root")
        h = read_header(path)
        metadata = h.get("__metadata__", {})
        require(metadata.get("pack_revision") == revision, "mixed shard revision")
        source_hash = metadata.get("source_sha256", "")
        require(re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None, "materialized source digest missing")
        entries = {k: v for k, v in h.items() if k != "__metadata__"}
        candidates = source_by_hash.get(source_hash, []) + source_by_hash.get("engram", [])
        matches = [(s, sh, r) for s, sh, r in candidates if
                   {k: v for k, v in sh.items() if k != "__metadata__"} == entries]
        require(len(matches) == 1, "materialized header not uniquely committed")
        source, sh, record = matches[0]
        require(source not in used_sources, "same source materialized twice")
        used_sources.add(source)
        materialized_sources[path.name] = source
        if record["kind"] == "engram":
            pin = record["state"]["pin"]
            require(pin["native_start"] == 0 and pin["rows"] == ENGRAM_ROWS.get(pin["layer"]) and pin["columns"] == 256,
                    "Engram manifest is sample or wrong geometry")
        if verify_payload:
            if record["kind"] == "engram":
                with Path(source).open("rb") as f:
                    data_start = 8 + int.from_bytes(f.read(8), "little")
                    for chunk in record["state"]["chunks"]:
                        for leaf, width in (("weight", 256), ("scale", 8)):
                            name = next(k for k in entries if k.endswith(".embed." + leaf))
                            offset = entries[name]["data_offsets"][0] + chunk["row"] * width
                            f.seek(data_start + offset)
                            raw = f.read(chunk["rows"] * width)
                            require(hashlib.sha256(raw).hexdigest() == chunk[leaf], "Engram committed chunk payload differs")
            require(sha(source) == source_hash, "source payload differs from committed materialization digest")
            require(sha(path, True) == sha(source, True), "materialized payload differs from source")
            payload_checks += 1
        headers[path.name] = h
        for key, info in entries.items():
            require(key not in all_tensors, "duplicate discovered tensor")
            all_tensors[key], owners[key] = info, path.name
    require(owners == mapping and used_sources == {str(Path(i["path"]).resolve()) for i in report["shards"]},
            "discovery/index/source coverage mismatch")
    require(not any(k.startswith("mtp.") for k in all_tensors), "unexpected DSpark tensors")
    actual_en, actual_tn = [], []
    routed_layers = {int(k.split(".")[2]) for k in all_tensors if k.startswith("model.layers.") and ".mlp.switch_mlp." in k}
    require(routed_layers == set(range(g["layers"])) and set(layer_manifests) == routed_layers,
            "routed layer coverage mismatch")
    for layer in range(g["layers"]):
        prefix = f"model.layers.{layer}.mlp.switch_mlp."
        banks = {k: v for k, v in all_tensors.items() if k.startswith(prefix)}
        groups = validate_routed(banks, eq, g["hidden"], g["intermediate"], g["experts"], g["top_k"])
        require(layer in layer_manifests, "layer permutation manifest missing")
        state = layer_manifests[layer]
        validate_permutation(state["expert_permutation"], g["experts"])
        layer_sources = {materialized_sources[owners[k]] for k in banks}
        require(len(layer_sources) == 1, "layer must have one authoritative assembled expert shard")
        committed_state = records[next(iter(layer_sources))]["state"]
        require(committed_state == state, "router permutation manifest is not authoritative shard owner")
        require(validate_quant(state["quantization"], "pair1") <= en and state["quantization"].get("tile_order") == eq.get("tile_order"), "layer manifest quantization differs")
        actual_en.extend(n for group in groups for n in group["n"])
        for suffix, shape in (("weight", [g["experts"], g["hidden"]]), ("bias", [g["experts"]]), ("bias_vl", [g["experts"]])):
            key = f"layers.{layer}.ffn.gate.{suffix}"
            require(key in all_tensors and all_tensors[key]["dtype"] in ("F32", "BF16") and all_tensors[key]["shape"] == shape,
                    "router trio geometry/dtype mismatch")
            require(owners[key] == owners[next(iter(banks))], "router is not owned by assembled expert shard")
        for filename in {owners[k] for k in banks}:
            validate_stamp(headers[filename].get("__metadata__", {}), eq, revision)
    for layer, rows in ENGRAM_ROWS.items():
        base = f"layers.{layer}.engram.embed."
        for suffix, dtype, shape in (("weight", "F8_E4M3", [rows, 256]), ("scale", "F8_E8M0", [rows, 8])):
            info = all_tensors.get(base + suffix)
            require(info is not None and info["dtype"] == dtype and info["shape"] == shape, "Engram table geometry/dtype mismatch")
        require(owners[base + "weight"] == owners[base + "scale"], "split Engram table")
        q, k = f"layers.{layer}.engram.q_weight", f"layers.{layer}.engram.k_weight"
        require(q in all_tensors and k in all_tensors and owners[q] == owners[k] and
                all_tensors[q]["dtype"] == all_tensors[k]["dtype"] == "F32" and
                all_tensors[q]["shape"] == all_tensors[k]["shape"] == [4, g["hidden"]], "Engram fused factors missing/split/wrong")
    spec1 = validate_trunk(all_tensors, tq)
    dense_trunk = global_count = 0
    for key, info in all_tensors.items():
        if ROUTED.fullmatch(key):
            continue
        if key.endswith(".trellis"):
            actual_tn.append(info["shape"][-1])
            validate_stamp(headers[owners[key]].get("__metadata__", {}), tq, revision)
        if key.startswith(("vision.", "aligner.", "image_")):
            state = records[materialized_sources[owners[key]]]["state"]
            donor_types = source_dtypes(key, state) if len(info["shape"]) == 1 else None
            validate_vision_dtype(key, info, donor_types)
        if not key.startswith(("layers.", "model.layers.")):
            global_count += 1
        elif key.startswith("layers.") and not (".ffn.gate." in key or ".engram.embed." in key or
                key.endswith((".trellis", ".suh", ".svh"))):
            dense_trunk += 1
    require(dict(spec1=spec1, dense_trunk=dense_trunk, globals=global_count) == counts, "actual tensor counts differ from explicit policy")
    require(max(actual_en) == en and actual_tn and max(actual_tn) == tn, "quant ceiling not widest actual bank")
    for key, shape in (("embed.weight", [g["vocab"], g["hidden"]]), ("head.weight", [g["vocab"], g["hidden"]]), ("norm.weight", [g["hidden"]])):
        require(key in all_tensors and all_tensors[key]["shape"] == shape, "global text tensor shape mismatch")
    for name, record in report["assets"].items():
        require((root / name).is_file(), "materialized asset missing")
        if name != "config.json":
            require(sha(root / name) == record["sha256"], "asset digest mismatch")
    return dict(status="runtime_unverified", structural_preflight="pass", ready=False,
                counts=counts, layers=g["layers"], experts_per_layer=g["experts"], revision=revision,
                payload_mode="verified_against_sources" if verify_payload else "committed_hashes_and_headers_only",
                payload_shards_verified=payload_checks,
                limitations=["Default mode trusts supplied committed hashes; it does not detect weight payload bitrot.",
                             "Router row data permutation and fused gate arithmetic require upstream evidence.",
                             "Asset equivalence, whole-pack budget, Sushi load and held-out quality remain external gates."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--report", required=True)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--expected-spec1", type=int, default=337)
    p.add_argument("--expected-dense-trunk", type=int, default=464)
    p.add_argument("--expected-globals", type=int, default=269)
    p.add_argument("--verify-payload", action="store_true")
    a = p.parse_args()
    result = check_candidate(a.root, a.report, a.manifests,
                             expected_counts=dict(spec1=a.expected_spec1, dense_trunk=a.expected_dense_trunk, globals=a.expected_globals),
                             verify_payload=a.verify_payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
