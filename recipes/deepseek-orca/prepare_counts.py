"""Generate bounded pinned imatrix count coverage; no weights downloaded."""
import argparse
import json
from pathlib import Path
from imatrix_reader import ImatrixReader

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');a=p.parse_args()
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    result=ImatrixReader().coverage()
    if result['projection_vectors']!=120 or result['unverified_vectors']:raise ValueError('Incomplete count coverage')
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('VERIFIED_COUNTS',result['projection_vectors'],'unvisited_projection_jobs',result['zero_projection_experts'])
