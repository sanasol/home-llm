"""Create donor globals and a separate F32-vector-corrected epoch, CPU only."""
import argparse
from pathlib import Path
from orca_reader import OrcaReader
from orca_export import write_globals
from fix_vision_precision import fix

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('original_output');p.add_argument('corrected_output')
    p.add_argument('--catalog',default=str(Path(__file__).with_name('orca_256g_tensors.json')));p.add_argument('--local-source-dir');a=p.parse_args()
    reader=OrcaReader(a.catalog,local_source_dir=a.local_source_dir)
    try:
        reader.add_vision();write_globals(reader,a.original_output)
        print(fix(a.original_output,a.corrected_output,reader))
    finally:reader.close()
