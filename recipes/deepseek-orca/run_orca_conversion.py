"""Bounded production orchestration; execution is opt-in, dry-run is default.

One spawn pool per physical GPU, CPU threads1; no GPU IPC and no full GGUF cache.
Work artifacts remain immutable after assembly. Lead owns all actual GPU launches.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path

PROJECTIONS = {"gate": "w1", "up": "w3", "down": "w2"}
RATES = (24, 28, 32, 40)
_WORKER = None


def selection(text, ceiling):
    values = set()
    for part in text.split(","):
        ends = part.split("-")
        if len(ends) == 1:
            values.add(int(part))
        elif len(ends) == 2:
            first, last = map(int, ends)
            if first > last:
                raise ValueError("Descending selection")
            values.update(range(first, last + 1))
        else:
            raise ValueError("Invalid selection")
    if not values or min(values) < 0 or max(values) >= ceiling:
        raise ValueError("Selection outside donor geometry")
    return sorted(values)


def job_plan(layers, experts, projections, n=28, policy=None):
    if n not in RATES or not projections or any(p not in PROJECTIONS for p in projections):
        raise ValueError("Invalid rate or projection")
    policy = policy or {}
    # Policy is exact layer -> expert -> projection -> n; no ambiguous global IDs.
    for layer, expert_policy in policy.items():
        if not layer.isdigit() or not 0 <= int(layer) < 40:
            raise ValueError("Invalid policy layer")
        for expert, rates in expert_policy.items():
            if not expert.isdigit() or not 0 <= int(expert) < 384:
                raise ValueError("Invalid policy expert")
            for p, rate in rates.items():
                if p not in PROJECTIONS or type(rate) is not int or rate not in RATES:
                    raise ValueError("Invalid policy projection rate")
    jobs = []
    for layer in layers:
        if type(layer) is not int or not 0 <= layer < 40:
            raise ValueError("Invalid layer")
        for expert in experts:
            if type(expert) is not int or not 0 <= expert < 384:
                raise ValueError("Invalid expert")
            for p in projections:
                rate = policy.get(str(layer), {}).get(str(expert), {}).get(p, n)
                jobs.append({"layer": layer, "expert": expert, "projection": p, "n": rate,
                             "id": f"L{layer:02d}-E{expert:03d}-{p}"})
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job")
    return jobs


def initialize_worker(gpu, settings):
    global _WORKER
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from orca_reader import OrcaReader
    from imatrix_reader import ImatrixReader
    reader = OrcaReader(settings["catalog"]) if settings.get("catalog") else OrcaReader()
    importance = ImatrixReader()
    identity = importance.identity()
    if identity != settings["imatrix_identity"]:
        raise ValueError("Worker calibration identity differs from pinned run")
    _WORKER = (reader, importance, settings)


def allow_unvisited_fallback(job, settings, error):
    return (settings["unvisited_policy"] == "identity-n28" and job["id"] in settings["unvisited_jobs"]
            and "Unvisited expert" in str(error) and job["n"] == 28)


def run_job(job):
    from orca_export import Manifest, atomic_json, projection, source_pin
    reader, importance, settings = _WORKER
    directory = Path(settings["output"]) / "jobs" / job["id"]
    params = {"job": job, "seed": settings["seed"], "backend": "cuda", "batch": settings["batch"],
              "run_policy_sha256": settings["policy_sha256"], "imatrix": settings["imatrix_identity"],
              "importance_normalization": "layer", "unvisited_policy": settings["unvisited_policy"],
              "imatrix_reader_sha256":settings["imatrix_reader_sha256"], "coverage_sha256":settings["coverage_sha256"]}
    manifest = Manifest(directory, source_pin(reader, params))
    native = f"layers.{job['layer']}.ffn.experts.{job['expert']}.{PROJECTIONS[job['projection']]}.weight"
    key = f"expert.{job['expert']}.{job['projection']}"
    if manifest.ready(key):
        if native not in manifest.data.get("importance_provenance", {}):
            raise ValueError("Completed artifact lacks required importance provenance")
        provenance = manifest.data["importance_provenance"][native]
        if provenance.get("status") == "unobserved-explicit-identity-fallback":
            manifest.data["complete"][key]["metrics"].update(calibrated=False,calibration_status=provenance["status"])
            atomic_json(manifest.path,manifest.data)
        path = directory / manifest.data["complete"][key]["file"]
        return {"job": job, "artifact": str(path), "manifest": str(manifest.path)}
    try:
        record = importance.read_native(native)
        hessian, provenance = record.weights("layer"), record.provenance
    except ValueError as error:
        if not allow_unvisited_fallback(job,settings,error):
            raise
        import numpy as np
        donor = reader.tensors[f"blk.{job['layer']}.ffn_{job['projection']}_exps.weight"]
        hessian = np.ones(donor.shape[-1], dtype=np.float64)
        provenance = {"status":"unobserved-explicit-identity-fallback", "normalization":"none",
                      "warning":"identity is not measured importance", "rate_n":28, "reason":str(error)}
    manifest.data["importance_provenance"] = {native: provenance}
    atomic_json(manifest.path,manifest.data)
    path = projection(reader, manifest, job["layer"], job["expert"], job["projection"], job["n"], params,hessian)
    if provenance.get("status") == "unobserved-explicit-identity-fallback":
        manifest.data["complete"][key]["metrics"].update(calibrated=False,calibration_status=provenance["status"])
    atomic_json(manifest.path, manifest.data)
    return {"job": job, "artifact": str(path), "manifest": str(manifest.path)}


def run_bounded(jobs, pools, callback, pending_per_pool):
    """Bound submission: no unbounded 46k-future queue."""
    iterator = iter(jobs)
    pending = {}
    def submit(pool):
        job = next(iterator, None)
        if job is None:
            return False
        pending[pool.submit(run_job, job)] = pool
        return True
    for pool in pools:
        for _ in range(pending_per_pool):
            if not submit(pool):
                break
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            pool = pending.pop(future)
            callback(future.result())
            submit(pool)


def finalize_layer(layer, jobs, run, reader, output):
    """Full384x3 only; retain all original job artifacts and checksum records."""
    import shutil
    import numpy as np
    from orca_export import Manifest, atomic_json, assemble_layer, digest, fused_engram, source_pin, write_safetensors
    from pair1_contract import PAIR1_CONFIG
    selected = [j for j in jobs if j["layer"] == layer]
    if {(j["expert"], j["projection"]) for j in selected} != {(e,p) for e in range(384) for p in PROJECTIONS}:
        raise ValueError("Layer finalization requires all384 experts and all3 projections")
    directory = Path(output) / "layers" / f"L{layer:02d}"
    import hashlib
    params = {"layer": layer, "policy_sha256": run.data["pin"]["policy_sha256"], "trunk": "dense-F32-pilot",
              "run_pin_sha256":hashlib.sha256(json.dumps(run.data["pin"],sort_keys=True).encode()).hexdigest(),
              "job_manifest_sha256":{j["id"]:digest(run.data["complete"][j["id"]]["manifest"]) for j in selected}}
    manifest = Manifest(directory, source_pin(reader, params))
    if manifest.ready("layer"):
        return str(directory / f"model-layer-{layer:02d}.safetensors")
    signatures = {}
    for j in selected:
        record = run.data["complete"].get(j["id"])
        if record is None:
            raise ValueError("Cannot assemble missing job")
        source = Path(record["artifact"])
        job_state = json.loads(Path(record["manifest"]).read_text())
        jp = job_state["pin"]["params"]
        root_params = run.data["pin"]["params"]
        if jp["job"] != j or jp["run_policy_sha256"] != root_params["policy_sha256"] or jp["imatrix"] != root_params["imatrix_identity"]:
            raise ValueError("Job pin differs from selected source/calibration policy")
        for field in ("repo","revision","catalog","encoder_sha256"):
            if job_state["pin"][field] != run.data["pin"][field]:
                raise ValueError("Job source/encoder pin differs from run")
        key = f"expert.{j['expert']}.{j['projection']}"
        expected = job_state["complete"][key]["sha256"]
        if digest(source) != expected:
            raise ValueError("Cannot assemble corrupt job artifact")
        destination = directory / source.name
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError("Existing layer staging artifact differs")
        else:
            try:
                os.link(source, destination)
            except OSError:
                temporary = destination.with_suffix(".tmp")
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
        signatures.setdefault(j["expert"], {})[j["projection"]] = j["n"]
    groups = {}
    for e in range(384):
        groups.setdefault(tuple(signatures[e][p] for p in PROJECTIONS), []).append(e)
    if len(groups) > 32:
        raise ValueError("Rate policy requires more than32 consumer groups")
    dense_files = []
    for native in sorted(reader.native_sources()):
        if not native.startswith(f"layers.{layer}.") or ".experts." in native:
            continue
        if native.endswith("engram.embed.weight"):
            manifest.data["external_requirements"] = [native, native[:-6] + "scale"]
            continue
        if native.endswith("engram.k_weight"):
            continue
        key = "dense." + native
        filename = key + ".safetensors"
        if not manifest.ready(key):
            if native.endswith("engram.q_weight"):
                q,k = fused_engram(reader.read(f"blk.{layer}.engram_gate.weight"))
                arrays = {native:q,native.replace("q_weight","k_weight"):k}
            else:
                arrays = {native:np.asarray(reader.read_native(native),np.float32)}
            write_safetensors(directory / filename, arrays, {"artifact":"driver-dense-trunk-pilot"})
            manifest.finish(key, filename)
        dense_files.append(directory / filename)
    assemble_layer(manifest,layer,sorted(groups.items()),dense_files,dict(PAIR1_CONFIG,k=max(j["n"] for j in selected)/16))
    atomic_json(manifest.path,manifest.data)
    return str(directory / f"model-layer-{layer:02d}.safetensors")


def output_lock(directory):
    import fcntl
    lock = (Path(directory) / ".conversion.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("Conversion output already owned by another driver")
    return lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--layers",default="0-39")
    p.add_argument("--experts",default="0-383")
    p.add_argument("--projections",default="gate,up,down")
    p.add_argument("--n",type=int,choices=RATES,default=28)
    p.add_argument("--rates",help="JSON layer->expert->projection->n")
    p.add_argument("--gpus",default="0",help="Physical GPU IDs, e.g.0,1")
    p.add_argument("--workers-per-gpu",type=int,default=1)
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--catalog")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--unvisited-policy",choices=("error","identity-n28"),default="error",help="Explicit count0 only fallback, flagged uncalibrated; never fabricated measured stats")
    p.add_argument("--counts-coverage",help="Existing ImatrixReader.coverage JSON; otherwise bounded live preflight")
    p.add_argument("--finalize-layers",action="store_true",help="After all jobs: dense-trunk pilot assembly, NOT final pack")
    a = p.parse_args()
    if a.workers_per_gpu < 1 or a.batch < 1:
        p.error("positive workers/batch required")
    gpu_ids = a.gpus.split(",")
    if any(not gpu.isdigit() for gpu in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        p.error("GPU IDs must be distinct nonnegative integers")
    policy = json.loads(Path(a.rates).read_text()) if a.rates else {}
    jobs = job_plan(selection(a.layers,40),selection(a.experts,384),a.projections.split(","),a.n,policy)
    output = Path(a.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    # Dry-run performs no network/torch imports and never starts a process pool.
    plan = {"status":"job-plan-not-executed","jobs":jobs,"policy":policy,"gpus":a.gpus,
            "workers_per_gpu":a.workers_per_gpu,"max_pending":2*a.workers_per_gpu*len(a.gpus.split(",")),
            "finalization":"full-layer dense-F32-trunk pilot only; no deletion"}
    if not a.execute:
        (output / "job-plan.json").write_text(json.dumps(plan,indent=2))
        print(json.dumps({"jobs":len(jobs),"status":plan["status"]}))
        return
    import hashlib
    from orca_export import Manifest, atomic_json, digest
    from orca_reader import OrcaReader
    from imatrix_reader import ImatrixReader
    with output_lock(output):
        importance = ImatrixReader()
        identity = importance.identity()
        coverage = json.loads(Path(a.counts_coverage).read_text()) if a.counts_coverage else importance.coverage()
        if coverage["source"] != identity or coverage["unverified_vectors"] or coverage["projection_vectors"] != 120:
            raise ValueError("Unverified or mismatched calibration coverage")
        unvisited = []
        for job in jobs:
            key = f"blk.{job['layer']}.ffn_{job['projection']}_exps.weight.counts"
            if job["expert"] in coverage["records"][key]["zero_experts"]:
                unvisited.append(job["id"])
                if a.unvisited_policy != "identity-n28" or job["n"] != 28:
                    raise ValueError("Selected unvisited expert requires explicit identity-n28 policy")
        atomic_json(output / "counts-coverage.json",coverage)
        settings = {"output":str(output),"catalog":a.catalog,"batch":a.batch,"seed":a.seed,"unvisited_policy":a.unvisited_policy,
                    "unvisited_jobs":unvisited,"imatrix_reader_sha256":digest(Path(__file__).with_name("imatrix_reader.py")),
                    "coverage_sha256":hashlib.sha256(json.dumps(coverage,sort_keys=True).encode()).hexdigest(),
                    "policy_sha256":hashlib.sha256(json.dumps({"jobs":jobs,"seed":a.seed,"batch":a.batch},sort_keys=True).encode()).hexdigest(),
                    "imatrix_identity":identity}
        from orca_export import source_pin
        reader = OrcaReader(a.catalog) if a.catalog else OrcaReader()
        pin = source_pin(reader,settings)
        pin.update(policy_sha256=settings["policy_sha256"],driver_sha256=digest(__file__))
        run = Manifest(output,pin)
        pools = []
        try:
            for gpu in a.gpus.split(","):
                if not gpu.isdigit():
                    raise ValueError("GPU IDs must be nonnegative integers")
                pools.append(ProcessPoolExecutor(max_workers=a.workers_per_gpu,mp_context=multiprocessing.get_context("spawn"),
                                                 initializer=initialize_worker,initargs=(gpu,settings)))
            def completed(record):
                run.data["complete"][record["job"]["id"]] = record
                atomic_json(run.path,run.data)
                print(json.dumps(record),flush=True)
            run_bounded(jobs,pools,completed,2*a.workers_per_gpu)
        finally:
            for pool in pools:
                pool.shutdown(wait=True,cancel_futures=True)
        if a.finalize_layers:
            for layer in sorted({j["layer"] for j in jobs}):
                print(finalize_layer(layer,jobs,run,reader,output),flush=True)



if __name__ == "__main__":
    main()
