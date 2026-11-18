"""Standalone calibrated donor trunk conversion, never visits routed experts.

One GPU/worker, immutable per-native artifacts. Final revision/index are deferred
to orca_export.materialize_pack. Dry-run is default; lead owns GPU execution.
"""
import argparse
import json
from pathlib import Path


def trunk_plan(reader, calibration_keys):
    selected, dense, excluded = [], [], []
    known = reader.native_sources()
    unknown = set(calibration_keys) - set(known)
    if unknown:
        raise ValueError(f"Calibration contains unmapped native keys: {sorted(unknown)}")
    for native in sorted(known):
        router = native.endswith(('ffn.gate.weight','ffn.gate.bias','ffn.gate.bias_vl'))
        if router and native in calibration_keys:
            raise ValueError("Router calibration belongs to expert assembly, not trunk")
        if router or not native.startswith("layers.") or ".experts." in native or ".engram.embed." in native:
            excluded.append(native)
            continue
        if native in calibration_keys:
            if not native.endswith('.weight') or native.endswith('ffn.gate.weight'):
                raise ValueError("Calibration must select trunk linear weights, not router")
            selected.append(native)
        else:
            dense.append(native)
    return {"spec1":selected,"dense":dense,"excluded":excluded,
            "status":"trunk-only-plan-not-executed","routed_experts":"never-read",
            "missing_calibration":"retain exact donor-dequantized F32, no invented stats"}


def convert_one(reader, manifest, native, hessian, settings):
    import numpy as np
    from orca_export import atomic_json, fused_engram, write_safetensors
    from pair1_contract import read_header, validate_trunk
    key = ('spec1.' if hessian is not None else 'dense.') + native
    filename = 'model-trunk-' + native + '.safetensors'
    if manifest.ready(key):
        return manifest.root / filename
    if '.experts.' in native or '.engram.embed.' in native or native.endswith(('ffn.gate.weight','ffn.gate.bias','ffn.gate.bias_vl')):
        raise ValueError("Trunk conversion cannot read routed experts, router trio, or huge Engram lookup")
    if native.endswith('engram.k_weight'):
        return None  # The compatible q work item co-locates both factors.
    if native.endswith('engram.q_weight'):
        layer = int(native.split('.')[1])
        q,k = fused_engram(reader.read(f'blk.{layer}.engram_gate.weight'))
        arrays = {native:q,native.replace('q_weight','k_weight'):k}
    else:
        arrays = {native:np.asarray(reader.read_native(native),dtype=np.float32)}
    if hessian is not None:
        from spec1_quality import encode_trunk
        w = arrays[native]
        h = np.asarray(hessian)
        if native.endswith('attn.wo_a.weight'):
            if h.ndim not in (2,3) or h.shape[0] != 8 or w.shape[0] != 8*1024:
                raise ValueError("wo_a needs explicit eight independent input Hessians")
            w = w.reshape(8,1024,w.shape[1]).transpose(0,2,1)
        else:
            w = w.T
        encoded = encode_trunk(w,h,n=settings['n'],seed=settings['seed'],name=native,
                               device=settings['device'],batch=settings['batch'])
        base = native[:-7]
        arrays = {base+'.'+part:encoded[part] for part in ('trellis','suh','svh')}
        metadata = {'format':'stq','codebook':'spec1','window':'15','k':str(settings['n']/16),
                    'status':'awaiting-whole-pack-finalizer'}
        metrics = {'calibrated':True,'calibration_loss':encoded['calibration_loss'],'n':settings['n']}
    else:
        metadata = {'status':'donor-F32-no-fabricated-calibration'}
        metrics = {'calibrated':False,'storage':'F32-exact-donor-dequantized'}
    write_safetensors(manifest.root / filename,arrays,metadata)
    if hessian is not None:
        validate_trunk(read_header(manifest.root / filename),dict(format='stq',codebook='spec1',window=15,
                       k=settings['n']/16,codebook_table='spec1-w15',codebook_digest='37f72431'))
    manifest.finish(key,filename,metrics)
    return manifest.root / filename


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('--calibration',required=True)
    p.add_argument('--provenance',required=True)
    p.add_argument('--catalog')
    p.add_argument('--layers',default='0-39')
    p.add_argument('--n',type=int,default=64,choices=range(16,113,16))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    if a.batch < 1 or a.gpu < 0:
        p.error('positive batch and nonnegative GPU required')
    import os
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import numpy as np
    from orca_reader import OrcaReader
    from run_orca_conversion import selection, output_lock
    from orca_export import Manifest, atomic_json, digest, source_pin
    reader = OrcaReader(a.catalog) if a.catalog else OrcaReader()
    layers = set(selection(a.layers,40))
    with np.load(a.calibration,allow_pickle=False) as hessians:
        plan = trunk_plan(reader,hessians.files)
        for kind in ('spec1','dense'):
            plan[kind] = [n for n in plan[kind] if int(n.split('.')[1]) in layers]
        root = Path(a.output).resolve()
        root.mkdir(parents=True,exist_ok=True)
        if not a.execute:
            atomic_json(root/'trunk-plan.json',plan)
            print(json.dumps({'spec1':len(plan['spec1']),'dense':len(plan['dense']),'status':plan['status']}))
            return
        with output_lock(root):
            import torch
            torch.set_num_threads(1)
            settings = {'n':a.n,'seed':a.seed,'device':'cuda','batch':a.batch,
                        'calibration_sha256':digest(a.calibration),'provenance_sha256':digest(a.provenance),
                        'layers':sorted(layers),'driver_sha256':digest(__file__)}
            manifest = Manifest(root,source_pin(reader,settings))
            manifest.data['plan'] = plan
            atomic_json(manifest.path,manifest.data)
            for native in plan['spec1'] + plan['dense']:
                h = hessians[native] if native in plan['spec1'] else None
                path = convert_one(reader,manifest,native,h,settings)
                if path:
                    print(str(path),flush=True)
            manifest.data['status'] = 'trunk-artifacts-complete-not-full-pack'
            atomic_json(manifest.path,manifest.data)


if __name__ == '__main__':
    main()
