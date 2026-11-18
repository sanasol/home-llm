"""Standalone trunk planning/serialization, CPU codecs mocked, no donor network."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_trunk_conversion as driver
from orca_export import Manifest
from pair1_contract import read_header


class Reader:
    def native_sources(self):
        return {n:['source'] for n in ('layers.0.attn.wq_a.weight','layers.0.ffn.gate.weight',
                'layers.0.engram.embed.weight','layers.0.ffn.experts.0.w2.weight','head.weight')}
    def read_native(self,native):
        assert '.experts.' not in native and '.engram.embed.' not in native
        return np.ones((128,128),np.float32)


class TrunkTests(unittest.TestCase):
    def test_plan_excludes_experts_huge_lookup_and_globals(self):
        plan=driver.trunk_plan(Reader(),['layers.0.attn.wq_a.weight'])
        self.assertEqual(plan['spec1'],['layers.0.attn.wq_a.weight'])
        self.assertEqual(plan['dense'],[])
        self.assertEqual(len(plan['excluded']),4)
        with self.assertRaises(ValueError):
            driver.trunk_plan(Reader(),['layers.0.ffn.gate.weight'])

    def test_fresh_spec1_rate_and_dense_resume(self):
        settings={'n':64,'seed':0,'device':'cpu','batch':1}
        with tempfile.TemporaryDirectory() as directory:
            manifest=Manifest(directory,{'source':'test'})
            def codec(w,h,**kw):
                self.assertEqual(kw['n'],64)
                self.assertEqual(w.shape,(128,128))
                return {'trellis':np.zeros((8,8,64),np.uint16),'suh':np.ones(128,np.float16),
                        'svh':np.ones(128,np.float16),'calibration_loss':1.5}
            with patch('spec1_quality.encode_trunk',side_effect=codec) as encode:
                path=driver.convert_one(Reader(),manifest,'layers.0.attn.wq_a.weight',np.ones(128),settings)
                header=read_header(path)
                self.assertEqual(header['layers.0.attn.wq_a.trellis']['shape'],[8,8,64])
                self.assertEqual(header['__metadata__']['codebook'],'spec1')
                driver.convert_one(Reader(),manifest,'layers.0.attn.wq_a.weight',np.ones(128),settings)
                self.assertEqual(encode.call_count,1)
            path=driver.convert_one(Reader(),manifest,'layers.0.attn.q_norm.weight',None,settings)
            self.assertEqual(read_header(path)['layers.0.attn.q_norm.weight']['dtype'],'F32')
            with self.assertRaisesRegex(ValueError,'cannot read'):
                driver.convert_one(Reader(),manifest,'layers.0.ffn.experts.0.w2.weight',None,settings)

    def test_trunk_expert_router_materialization_nonoverlap(self):
        from orca_export import conversion_plan, index_shards, write_safetensors
        from orca_reader import Tensor
        class CombinedReader:
            tensors={'blk.0.ffn_gate_inp.weight':Tensor('x',(128,2),'F32',0,0),
                     'blk.0.attn_q_a_norm.weight':Tensor('x',(128,),'F32',1024,0)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=Manifest(root/'trunk',{'source':'test'})
            trunk=driver.convert_one(Reader(),manifest,'layers.0.attn.q_norm.weight',None,{'n':64})
            router=root/'expert-router.safetensors'
            write_safetensors(router,{'layers.0.ffn.gate.weight':np.ones((2,128),np.float32)}, {})
            report=index_shards(conversion_plan(CombinedReader()),[trunk,router],root/'index.json')
            self.assertEqual(report['coverage']['missing'],[])
            self.assertEqual(len(report['weight_map']),2)
            with self.assertRaisesRegex(ValueError,'Duplicate destination'):
                index_shards(conversion_plan(CombinedReader()),[router,router],root/'bad.json')

    def test_grouped_wo_a_explicit_orientation(self):
        class GroupReader:
            def read_native(self,native):
                return np.zeros((8192,4096),np.float32)
        settings={'n':64,'seed':0,'device':'cpu','batch':1}
        with tempfile.TemporaryDirectory() as directory:
            manifest=Manifest(directory,{'source':'test'})
            def codec(w,h,**kw):
                self.assertEqual(w.shape,(8,4096,1024))
                self.assertEqual(h.shape,(8,4096))
                return {'trellis':np.zeros((8,256,64,64),np.uint16),
                        'suh':np.ones((8,4096),np.float16),'svh':np.ones((8,1024),np.float16),'calibration_loss':0.}
            with patch('spec1_quality.encode_trunk',side_effect=codec):
                path=driver.convert_one(GroupReader(),manifest,'layers.0.attn.wo_a.weight',np.ones((8,4096)),settings)
                self.assertEqual(read_header(path)['layers.0.attn.wo_a.trellis']['shape'],[8,256,64,64])


if __name__=='__main__':
    unittest.main()
