"""CPU miniature assembly; real geometry is independently contract-checked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import assemble_orca_experts as assembler
from orca_export import Manifest,write_safetensors
import pair1_contract as contract


class AssemblyTests(unittest.TestCase):
    def fixture(self,root):
        pin={'repo':'donor','revision':'revision','catalog':{},'encoder_sha256':{'encoder':'hash'},
             'params':{'policy_sha256':'policy','imatrix_identity':{'header':'hash'}}}
        run={'pin':pin,'complete':{}}
        for e,n in enumerate((32,24)):
            for p in ('gate','up','down'):
                job={'layer':0,'expert':e,'projection':p,'n':n,'id':f'L00-E{e:03d}-{p}'}
                jobpin=dict(pin,params={'job':job,'run_policy_sha256':'policy','imatrix':{'header':'hash'}})
                manifest=Manifest(root/job['id'],jobpin)
                key=f'expert.{e}.{p}'
                path=manifest.root/(key+'.safetensors')
                write_safetensors(path,{'trellis':np.full((8,8,n),e,np.uint16),
                    'suh':np.ones(128,np.float16),'svh':np.ones(128,np.float16)}, {})
                native=f'layers.0.ffn.experts.{e}.{dict(gate="w1",up="w3",down="w2")[p]}.weight'
                manifest.data['importance_provenance']={native:{'status':'measured'}}
                manifest.finish(key,path.name,{'calibrated':True})
                run['complete'][job['id']]={'job':job,'artifact':str(path),'manifest':str(manifest.path)}
        return run

    def test_banks_and_permuted_router_only(self):
        class Reader:
            def read_native(self,name):
                if name.endswith('weight'):
                    return np.array([[1]*128,[2]*128],np.float32)
                return np.array([3,4],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.fixture(root/'jobs')
            original=contract.validate_routed
            def sourcepin(reader,params):
                return dict(run['pin'],params=params)
            with patch('orca_export.source_pin',side_effect=sourcepin),patch('pair1_contract.validate_routed',
                       side_effect=lambda h,q,i,o,e:original(h,q,i,o,e,top_k=2)):
                path=assembler.assemble(Reader(),run,0,root/'assembled',experts=2)
            header=contract.read_header(path)
            self.assertEqual(len(header)-1,21) # two rate groups * nine banks + router trio.
            arrays=contract.read_tensors(path)
            np.testing.assert_array_equal(arrays['layers.0.ffn.gate.bias'],[4,3])
            self.assertTrue(all('.switch_mlp.' in k or '.ffn.gate.' in k for k in arrays))
            source=next(iter(run['complete'].values()))['artifact']
            with Path(source).open('ab') as f:f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                assembler.committed_jobs(run,0,2)

    def test_full_layer_required_and_source_rate_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                assembler.committed_jobs(run,0) # production requires384*3.
            job=next(iter(run['complete'].values()))
            job['job']['n']=40
            with self.assertRaisesRegex(ValueError,'policy differs'):
                assembler.committed_jobs(run,0,2)


if __name__=='__main__':
    unittest.main()
