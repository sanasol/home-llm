"""New globals epoch: retain F32 vision vectors, preserve strict BF16 matrices.

Never changes old269 committed files. Unchanged items use hardlinks; only donor
F32 native vectors are fetched again. Matrix/patch weights remain BF16 as required
by Sushi's strict vision linear loader. No GPU or full global re-download.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

IMAGE = {'image_start','image_end','image_newline'}


def needs_f32(native, info, sources, tensors):
    vision = native.startswith(('vision.','aligner.')) or native in IMAGE
    original_f32 = all(tensors[source].quant=='F32' for source in sources)
    return vision and len(info['shape'])==1 and original_f32


def validate_precision(native,info):
    vision=native.startswith(('vision.','aligner.')) or native in IMAGE
    if not vision:
        return
    if len(info['shape'])>=2 and info['dtype']!='BF16':
        raise ValueError('Vision matrix/patch requires BF16')
    if len(info['shape'])==1 and info['dtype']!='F32':
        raise ValueError('Pinned donor vision vector must retain F32')


def fix(old_directory,new_directory,reader):
    from orca_export import Manifest,atomic_json,digest,source_pin,write_safetensors
    from pair1_contract import read_header
    from run_orca_conversion import output_lock
    old,new=Path(old_directory).resolve(),Path(new_directory).resolve()
    if old==new or old in new.parents or new in old.parents:
        raise ValueError('Globals precision epochs must be disjoint')
    original_path=old/'manifest.json'
    original=json.loads(original_path.read_text())
    original_sha=digest(original_path)
    current=source_pin(reader,{})
    for field in ('repo','revision','catalog'):
        if current[field]!=original['pin'][field]:
            raise ValueError('Globals donor source/catalog differs')
    sources=reader.native_sources()
    expected={'global.'+name for name in sources if not name.startswith('layers.')}
    if set(original['complete'])!=expected:
        raise ValueError('Old globals inventory incomplete or unexpected')
    new.mkdir(parents=True,exist_ok=True)
    with output_lock(new):
        params={'mode':'globals-precision-v2','old_globals_manifest_sha256':original_sha,
                'policy':'F32 donor vectors retained; BF16 vision matrices preserved',
                'fixer_sha256':digest(__file__)}
        manifest=Manifest(new,source_pin(reader,params))
        for key,record in sorted(original['complete'].items()):
            native=key[len('global.'):]
            source=old/record['file']
            if Path(record['file']).name!=record['file'] or source.resolve().parent!=old or digest(source)!=record['sha256']:
                raise ValueError('Old globals artifact ownership/checksum mismatch')
            header=read_header(source)
            if set(header)-{'__metadata__'}!={native}:
                raise ValueError('Old globals artifact native identity mismatch')
            filename=record['file']
            target=new/filename
            if manifest.ready(key):
                continue
            corrected=needs_f32(native,header[native],sources[native],reader.tensors)
            if corrected:
                import numpy as np
                values=np.asarray(reader.read_native(native),dtype=np.float32)
                if list(values.shape)!=header[native]['shape'] or not np.isfinite(values).all():
                    raise ValueError('Donor correction shape/nonfinite mismatch')
                write_safetensors(target,{native:values},{'artifact':'donor-F32-vector-precision-correction'})
            else:
                if target.exists():
                    if digest(target)!=record['sha256']:
                        raise ValueError('Existing unchanged global artifact differs')
                else:
                    temporary=target.with_suffix(target.suffix+'.tmp')
                    if temporary.exists():
                        if temporary.resolve().parent!=new or temporary.is_symlink():
                            raise ValueError('Unsafe globals import temporary')
                        if digest(temporary)!=record['sha256']:
                            temporary.unlink()
                    if not temporary.exists():
                        try:os.link(source,temporary)
                        except OSError:
                            with source.open('rb') as inp,temporary.open('xb') as out:
                                shutil.copyfileobj(inp,out,8<<20)
                                out.flush();os.fsync(out.fileno())
                    os.replace(temporary,target)
            validate_precision(native,read_header(target)[native])
            manifest.data.setdefault('precision_lineage',{})[native]={
                'old_globals_manifest_sha256':original_sha,'old_artifact_sha256':record['sha256'],
                'policy_change':'original-F32-vector-restored' if corrected else 'unchanged-byte-identical'}
            manifest.finish(key,filename,{'corrected_F32_vector':corrected})
        if digest(original_path)!=original_sha:
            raise ValueError('Old globals manifest changed during correction')
        manifest.data['status']='globals-precision-corrected-not-full-pack'
        atomic_json(manifest.path,manifest.data)
        return {'items':len(manifest.data['complete']),
                'corrected':sum(r['metrics']['corrected_F32_vector'] for r in manifest.data['complete'].values()),
                'new_root':str(new)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('old_root');p.add_argument('new_root');p.add_argument('--catalog')
    a=p.parse_args()
    from orca_reader import OrcaReader
    reader=OrcaReader(a.catalog) if a.catalog else OrcaReader()
    reader.add_vision()
    print(json.dumps(fix(a.old_root,a.new_root,reader),indent=2))


if __name__=='__main__':main()
