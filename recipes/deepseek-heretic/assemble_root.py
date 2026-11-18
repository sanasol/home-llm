"""Fetch verified pinned official root, then assemble a separate Heretic root.

Never modifies the official source. Every unchanged file is an immutable hardlink;
replacement trunks are independent copies, verified to change only trellis bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from hub import api, rng
from build import OFF, OFF_REV


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    entries = api(f'models/{OFF}/tree/{OFF_REV}?recursive=1&limit=1000')
    entries = [x for x in entries if x.get('type') == 'file' and '/' not in x['path']]
    if not entries or not any(x['path'] == 'model.safetensors.index.json' for x in entries):
        raise ValueError('Incomplete upstream root listing')
    def fetch(info):
        name, size = info['path'], info['size']
        target = root / name
        expected = info.get('lfs', {}).get('oid')
        if target.exists() and target.stat().st_size == size and expected and digest(target) == expected:
            return name, {'bytes': size, 'sha256': expected}
        if shutil.disk_usage(root).free < size + 8 * 1024**3:
            raise RuntimeError('Keep an 8 GiB reserve beyond pending files')
        temporary = root / (name + '.partial')
        request_url = f'https://huggingface.co/{OFF}/resolve/{OFF_REV}/{name}'
        from hub import _req, urlopen
        h = hashlib.sha256()
        with urlopen(_req(request_url), timeout=300) as response, temporary.open('wb') as f:
            remaining = size
            while remaining:
                block = response.read(min(8 << 20, remaining))
                if not block:
                    raise EOFError('Short pinned official download')
                f.write(block); h.update(block); remaining -= len(block)
            if response.read(1):
                raise ValueError('Oversized upstream response')
        if expected and h.hexdigest() != expected:
            raise ValueError('Upstream LFS checksum mismatch: ' + name)
        # Small Git objects are bound by immutable source revision and Git blob SHA1.
        if not expected:
            blob = hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + temporary.read_bytes()).hexdigest()
            if blob != info['oid']:
                raise ValueError('Upstream Git blob mismatch: ' + name)
        temporary.replace(target)
        print('VERIFIED', name, flush=True)
        return name, {'bytes': size, 'sha256': h.hexdigest()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = dict(pool.map(fetch, entries))
    (root / 'OFFICIAL_VERIFIED.json').write_text(json.dumps({'repo': OFF, 'revision': OFF_REV, 'files': records}, indent=2) + '\n')


def verify_overlay(actual, original, layer):
    n = struct.unpack_from('<Q', original)[0]; start = 8 + n
    if len(actual) != len(original) or actual[:start] != original[:start]:
        raise ValueError('Replacement header/size changed')
    header = json.loads(original[8:start]); info = header[f'layers.{layer}.attn.wo_b.trellis']
    a, b = info['data_offsets']
    if info['dtype'] != 'U16' or info['shape'] != [512, 320, 64] or b-a != 512*320*64*2:
        raise ValueError('Unexpected edited trellis geometry')
    if actual[:start+a] != original[:start+a] or actual[start+b:] != original[start+b:]:
        raise ValueError('Non-trellis bytes changed')
    if actual[start+a:start+b] == original[start+a:start+b]:
        raise ValueError('Edited trellis did not change')


def assemble(base, overlay, output):
    base, overlay, output = map(lambda p: Path(p).resolve(), (base, overlay, output))
    if output.exists() or output == base or base in output.parents or output in base.parents or output == overlay or overlay in output.parents or output in overlay.parents:
        raise ValueError('Require a fresh output disjoint from the official root')
    manifest = json.loads((base / 'OFFICIAL_VERIFIED.json').read_text())
    if (manifest['repo'], manifest['revision']) != (OFF, OFF_REV):
        raise ValueError('Official provenance mismatch')
    expected = {f'model-trunk-{i:02d}.safetensors' for i in range(2, 40)}
    if {p.name for p in overlay.glob('*.safetensors')} != expected:
        raise ValueError('Exactly 38 replacement trunk shards required')
    for name, record in manifest['files'].items():
        source = base / name
        if source.stat().st_size != record['bytes'] or digest(source) != record['sha256']:
            raise ValueError('Official source changed: ' + name)
    for layer in range(2, 40):
        name = f'model-trunk-{layer:02d}.safetensors'
        verify_overlay((overlay / name).read_bytes(), (base / name).read_bytes(), layer)
    output.mkdir()
    for name in manifest['files']:
        if name in expected:
            shutil.copyfile(overlay / name, output / name)
        else:
            os.link(base / name, output / name)
    provenance = {'official_repo': OFF, 'official_revision': OFF_REV, 'changed': sorted(expected),
                  'scope': '38 attention wo_b trellis banks only; official experts/scales/header bytes retained',
                  'method': 'F32 uncalibrated SPEC1 least-squares, not private Sashimi',
                  'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in output.iterdir()}}
    (output / 'HERETIC_BUILD_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('download'); a.add_argument('base')
    a = sub.add_parser('assemble'); a.add_argument('base'); a.add_argument('overlay'); a.add_argument('output')
    args = p.parse_args()
    if args.command == 'download': download(Path(args.base))
    else: assemble(args.base, args.overlay, args.output)
