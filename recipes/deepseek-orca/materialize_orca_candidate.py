"""Explicit committed-artifact final candidate planner; no GPU or auto execution.

Dependencies: orca_export.py/materialize_pack, check_full_candidate.py and their
local imports. Never passes expert staging or per-job files to the finalizer.
Full completeness is required BEFORE any multi-hundred-GB payload rewrite.
"""
import argparse
import json
from pathlib import Path
import shutil


def inventory(experts_root,trunk_root,globals_root,engram_paths,*,layers=40,counts=None):
    from orca_export import digest
    counts=counts or {'spec1':337,'dense_trunk':464,'globals':269}
    shards=[];manifests=[];missing=[]
    actual={'experts':0,'spec1':0,'dense_trunk':0,'globals':0,'engram':0}
    def committed(state_path,key,record):
        path=state_path.parent/record['file']
        if Path(record['file']).name!=record['file'] or path.resolve().parent!=state_path.parent.resolve():
            raise ValueError('Committed source escapes manifest directory')
        if not path.is_file():
            missing.append(str(path));return
        if digest(path)!=record['sha256']:
            raise ValueError('Committed source checksum mismatch: '+str(path))
        shards.append(str(path.resolve()))
    for layer in range(layers):
        state_path=Path(experts_root)/f'L{layer:02d}'/'manifest.json'
        if not state_path.is_file():missing.append(str(state_path));continue
        state=json.loads(state_path.read_text());record=state['complete'].get('layer')
        if record is None:missing.append(f'expert-layer-{layer:02d}');continue
        if state['pin']['params']['layer']!=layer:raise ValueError('Expert layer manifest mismatch')
        committed(state_path,'layer',record);actual['experts']+=1;manifests.append(str(state_path.resolve()))
    for root,kind in ((Path(trunk_root),'trunk'),(Path(globals_root),'globals')):
        state_path=root/'manifest.json'
        if not state_path.is_file():missing.append(str(state_path));continue
        state=json.loads(state_path.read_text());manifests.append(str(state_path.resolve()))
        for key,record in state['complete'].items():
            if kind=='trunk' and key.startswith('spec1.'):category='spec1'
            elif kind=='trunk' and key.startswith('dense.'):category='dense_trunk'
            elif kind=='globals' and key.startswith('global.'):category='globals'
            else:raise ValueError('Unexpected source work-item kind')
            committed(state_path,key,record)
            if category=='dense_trunk':
                from pair1_contract import read_header
                artifact=state_path.parent/record['file']
                if artifact.is_file():actual[category]+=len(set(read_header(artifact))-{'__metadata__'})
            else:actual[category]+=1
    engram_layers=set()
    for raw in engram_paths:
        path=Path(raw).resolve();state_path=path.with_suffix(path.suffix+'.resume.json')
        if not state_path.is_file():missing.append(str(state_path));continue
        state=json.loads(state_path.read_text())
        if not state.get('done') or not path.is_file():missing.append(str(path));continue
        pin=state['pin'];layer=pin['layer']
        expected={1:384006168,14:384016682}
        if layer in engram_layers:raise ValueError('Duplicate Engram layer')
        engram_layers.add(layer)
        if layer not in expected or pin['native_start']!=0 or pin['rows']!=expected[layer]:
            raise ValueError('Partial/sample Engram cannot enter final candidate')
        # Resume verifies committed range hashes on conversion; strict final checker
        # receives this resume manifest. Do not invent a full-file source checksum.
        if path.stat().st_size!=state['output_bytes']:raise ValueError('Engram output size changed')
        from check_full_candidate import committed_sources
        committed_sources([state_path])
        shards.append(str(path));manifests.append(str(state_path));actual['engram']+=1
    expected=dict(counts,experts=layers,engram=2)
    for key,value in expected.items():
        if actual[key]!=value:missing.append(f'{key}: expected {value}, found {actual[key]}')
    if len(set(shards))!=len(shards):raise ValueError('Duplicate artifact path')
    return {'complete':not missing,'actual':actual,'expected':expected,'missing':missing,
            'shards':shards,'manifests':manifests,'bytes':sum(Path(p).stat().st_size for p in shards)}


def preflight_sources(reader,report,assets,output):
    """Bounded header/coverage validation BEFORE any candidate payload copies."""
    from orca_export import conversion_plan,index_shards
    from pair1_contract import read_header,validate_trunk
    from check_full_candidate import committed_sources,source_dtypes,validate_vision_dtype
    records,_=committed_sources(report['manifests'])
    coverage=index_shards(conversion_plan(reader),report['shards'],output,assets)
    if coverage['coverage']['missing'] or coverage['coverage']['unsupported_sources'] or coverage['missing_assets']:
        raise ValueError('Source header/name/asset coverage incomplete before materialization')
    all_tensors={}
    trunk_rates=[]
    for raw in report['shards']:
        path=str(Path(raw).resolve());header=read_header(path)
        if records[path]['kind']=='engram':
            import hashlib
            state=records[path]['state']
            with Path(path).open('rb') as f:
                start=8+int.from_bytes(f.read(8),'little')
                for chunk in state['chunks']:
                    for leaf,width in (('weight',256),('scale',8)):
                        name=next(k for k in header if k.endswith('.embed.'+leaf))
                        offset=start+header[name]['data_offsets'][0]+chunk['row']*width
                        f.seek(offset);remaining=chunk['rows']*width;h=hashlib.sha256()
                        while remaining:
                            block=f.read(min(remaining,8<<20))
                            if not block:raise ValueError('Truncated Engram committed source')
                            h.update(block);remaining-=len(block)
                        if h.hexdigest()!=chunk[leaf]:raise ValueError('Engram committed source chunk changed')
        record=records[path];state=record['state']
        if 'expert_permutation' in state:
            layer=state['pin']['params']['layer']
            if any(not ('.switch_mlp.' in k or k.startswith(f'layers.{layer}.ffn.gate.')) for k in header if k!='__metadata__'):
                raise ValueError('Expert source contains dense trunk; experts-only assembly required')
        if any(k.startswith(('vision.','aligner.','image_')) for k in header):
            if 'precision_lineage' not in state:
                raise ValueError('Vision source requires corrected globals-v2 precision lineage')
        if state['pin']['repo']!=__import__('orca_reader').REPO or state['pin']['revision']!=__import__('orca_reader').REVISION:
            raise ValueError('Source manifest donor revision differs')
        for name,info in header.items():
            if name=='__metadata__':continue
            if name in all_tensors:raise ValueError('Duplicate source tensor')
            all_tensors[name]=info
            if name.endswith('.trellis') and '.switch_mlp.' not in name:trunk_rates.append(info['shape'][-1])
            if name.startswith(('vision.','aligner.','image_')):
                original=source_dtypes(name,state) if len(info['shape'])==1 else None
                validate_vision_dtype(name,info,original)
    if trunk_rates:
        validate_trunk(all_tensors,dict(format='stq',codebook='spec1',window=15,k=max(trunk_rates)/16,
                       codebook_table='spec1-w15',codebook_digest='37f72431'))
    return coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('--experts-root',required=True);p.add_argument('--trunk-root',required=True)
    p.add_argument('--globals-root',required=True);p.add_argument('--engram',nargs=2,required=True)
    p.add_argument('--assets',required=True);p.add_argument('--catalog')
    p.add_argument('--execute',action='store_true');p.add_argument('--reserve-gib',type=int,default=20)
    a=p.parse_args()
    from orca_export import atomic_json,conversion_plan,materialize_pack
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=inventory(a.experts_root,a.trunk_root,a.globals_root,a.engram)
    atomic_json(output/'materialization-plan.json',report)
    print(json.dumps({k:report[k] for k in ('complete','actual','expected','missing','bytes')},indent=2))
    if not a.execute:return
    if not report['complete']:raise ValueError('Refusing materialization of incomplete candidate')
    available=shutil.disk_usage(output).free
    required=report['bytes']+a.reserve_gib*2**30
    if available<required:raise ValueError(f'Insufficient disk before payload rewrite: need {required}, free {available}')
    from orca_reader import OrcaReader
    from run_orca_conversion import output_lock
    from check_full_candidate import check_candidate
    reader=OrcaReader(a.catalog) if a.catalog else OrcaReader();reader.add_vision()
    assets=json.loads(Path(a.assets).read_text())
    with output_lock(output):
        preflight_sources(reader,report,assets,output/'source-preflight.json')
        materialize_pack(conversion_plan(reader),report['shards'],output,assets)
        checked=check_candidate(output,output/'coverage.json',report['manifests'])
        atomic_json(output/'strict-check.json',checked)
    print('Materialized and structurally checked; runtime/KLD/behavioral gates remain separate')


if __name__=='__main__':main()
