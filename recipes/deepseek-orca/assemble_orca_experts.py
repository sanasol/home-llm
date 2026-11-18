"""Offline experts-only assembly from committed driver job manifests.

Reads a root run-manifest snapshot atomically; writes a DIFFERENT locked root.
Routed banks and identically permuted router trio only, never dense trunk/lookup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def committed_jobs(run, layer, experts=384):
    """Resolve exact job set from committed results; verify bytes and source pins."""
    from orca_export import digest
    from run_orca_conversion import PROJECTIONS
    records = []
    root_params = run['pin']['params']
    for expert in range(experts):
        for projection in PROJECTIONS:
            job_id = f'L{layer:02d}-E{expert:03d}-{projection}'
            record = run['complete'].get(job_id)
            if record is None:
                raise ValueError(f'Layer incomplete: {job_id}')
            job = record['job']
            if (job['id'],job['layer'],job['expert'],job['projection']) != (job_id,layer,expert,projection):
                raise ValueError('Committed job identity mismatch')
            manifest_path = Path(record['manifest']).resolve()
            state = json.loads(manifest_path.read_text())
            params = state['pin']['params']
            if params['job'] != job or params['run_policy_sha256'] != root_params['policy_sha256'] or params['imatrix'] != root_params['imatrix_identity']:
                raise ValueError('Committed job calibration/rate policy differs from root')
            for field in ('repo','revision','catalog','encoder_sha256'):
                if state['pin'][field] != run['pin'][field]:
                    raise ValueError('Committed job source/encoder differs from root')
            key = f'expert.{expert}.{projection}'
            artifact = Path(record['artifact']).resolve()
            stored = state['complete'][key]
            if artifact != manifest_path.parent / stored['file']:
                raise ValueError('Artifact does not match job manifest ownership')
            if digest(artifact) != stored['sha256']:
                raise ValueError('Committed expert artifact checksum mismatch')
            native = f'layers.{layer}.ffn.experts.{expert}.{PROJECTIONS[projection]}.weight'
            if native not in state.get('importance_provenance',{}):
                raise ValueError('Committed expert lacks importance provenance')
            provenance = state['importance_provenance'][native]
            if provenance.get('status') == 'unobserved-explicit-identity-fallback' and stored['metrics'].get('calibrated') is not False:
                raise ValueError('Identity fallback artifact falsely marked calibrated')
            records.append({'job':job,'artifact':str(artifact),'manifest_sha256':digest(manifest_path),
                            'artifact_sha256':stored['sha256'],'provenance':provenance})
    return records


def assemble(reader, run, layer, output, *, experts=384):
    from orca_export import Manifest, assemble_layer, digest, source_pin, write_safetensors
    from pair1_contract import PAIR1_CONFIG, read_header
    records = committed_jobs(run,layer,experts)
    # This separate assembler may use a newer reader implementation, but only the
    # identical pinned donor revision/catalog; encoder identity stays original.
    current = source_pin(reader,{})
    for field in ('repo','revision','catalog'):
        if current[field] != run['pin'][field]:
            raise ValueError('Assembler donor source differs from committed run')
    params = {'layer':layer,'run_pin_sha256':hashlib.sha256(json.dumps(run['pin'],sort_keys=True).encode()).hexdigest(),
              'jobs':records,'assembler_sha256':digest(__file__),'content':'experts-and-router-only'}
    manifest = Manifest(Path(output)/f'L{layer:02d}',source_pin(reader,params))
    if manifest.ready('layer'):
        return manifest.root/f'model-layer-{layer:02d}.safetensors'
    rates = {}
    for record in records:
        job = record['job']
        key = f"expert.{job['expert']}.{job['projection']}"
        source = Path(record['artifact'])
        header = read_header(source)
        if set(header)-{'__metadata__'} != {'trellis','suh','svh'} or header['trellis']['shape'][-1] != job['n']:
            raise ValueError('Projection artifact bank/rate mismatch')
        target = manifest.root/(key+'.safetensors')
        if target.exists():
            if digest(target) != record['artifact_sha256']:
                raise ValueError('Existing staged expert differs')
        else:
            try:
                os.link(source,target)
            except OSError:
                temporary = target.with_suffix('.tmp')
                shutil.copyfile(source,temporary)
                os.replace(temporary,target)
        rates.setdefault(job['expert'],{})[job['projection']] = job['n']
    groups = {}
    for expert in range(experts):
        groups.setdefault(tuple(rates[expert][p] for p in ('gate','up','down')),[]).append(expert)
    if len(groups)>32:
        raise ValueError('More than32 routed rate groups')
    routers = []
    for suffix in ('weight','bias','bias_vl'):
        native=f'layers.{layer}.ffn.gate.{suffix}'
        source=manifest.root/f'router.{suffix}.safetensors'
        key='router.'+suffix
        if not manifest.ready(key):
            import numpy as np
            value=np.asarray(reader.read_native(native),dtype=np.float32)
            if value.shape[0] != experts:
                raise ValueError('Router expert count differs from committed banks')
            write_safetensors(source,{native:value},{'source':native,'content':'router-only'})
            manifest.finish(key,source.name)
        routers.append(source)
    # Existing layout oracle requires full real top6; miniature tests explicitly
    # inject a test-only top-k wrapper without weakening production validation.
    quant=dict(PAIR1_CONFIG,k=max(r['job']['n'] for r in records)/16)
    assemble_layer(manifest,layer,sorted(groups.items()),routers,quant)
    result=manifest.root/f'model-layer-{layer:02d}.safetensors'
    header=read_header(result)
    if any(not ('.mlp.switch_mlp.' in key or key.startswith(f'layers.{layer}.ffn.gate.')) for key in header if key!='__metadata__'):
        raise ValueError('Experts-only assembly unexpectedly contains trunk')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_manifest')
    p.add_argument('output')
    p.add_argument('--layers',default='0-39')
    p.add_argument('--catalog')
    a=p.parse_args()
    from run_orca_conversion import selection,output_lock
    from orca_reader import OrcaReader
    source=Path(a.run_manifest).resolve()
    output=Path(a.output).resolve()
    if output==source.parent or source.parent in output.parents:
        p.error('Offline output must be separate from live production run root')
    output.mkdir(parents=True,exist_ok=True)
    reader=OrcaReader(a.catalog) if a.catalog else OrcaReader()
    with output_lock(output):
        run=json.loads(source.read_text())
        for layer in selection(a.layers,40):
            print(assemble(reader,run,layer,output),flush=True)


if __name__=='__main__':
    main()
