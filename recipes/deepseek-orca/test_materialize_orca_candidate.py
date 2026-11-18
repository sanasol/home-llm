"""Tiny committed inventory tests; no GPU, network or final large copies."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import materialize_orca_candidate as runner
from orca_export import Manifest,atomic_json,write_safetensors


class CandidateTests(unittest.TestCase):
    def fixture(self,root):
        experts=root/'experts';trunk=root/'trunk';glob=root/'globals'
        e=Manifest(experts/'L00',{'params':{'layer':0}})
        write_safetensors(e.root/'model-layer-00.safetensors',{'expert':np.ones(1,np.float32)}, {})
        e.finish('layer','model-layer-00.safetensors')
        t=Manifest(trunk,{'test':True})
        write_safetensors(trunk/'spec.safetensors',{'trellis':np.ones(1,np.uint16)}, {})
        t.finish('spec1.bank','spec.safetensors')
        write_safetensors(trunk/'factors.safetensors',{'q':np.ones(1,np.float32),'k':np.ones(1,np.float32)}, {})
        t.finish('dense.q','factors.safetensors')
        g=Manifest(glob,{'test':True})
        write_safetensors(glob/'global.safetensors',{'global':np.ones(1,np.float32)}, {})
        g.finish('global.one','global.safetensors')
        engrams=[]
        for layer,rows in ((1,384006168),(14,384016682)):
            path=root/f'engram-{layer}.safetensors';path.write_bytes(b'tiny')
            atomic_json(path.with_suffix('.safetensors.resume.json'),{'done':True,'pin':{'layer':layer,'native_start':0,'rows':rows},'output_bytes':4,'chunks':[{'row':0,'rows':rows,'weight':'0'*64,'scale':'1'*64}]})
            engrams.append(path)
        return experts,trunk,glob,engrams

    def test_exact_inventory_does_not_include_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=self.fixture(Path(directory))
            report=runner.inventory(*roots,layers=1,counts={'spec1':1,'dense_trunk':2,'globals':1})
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['shards']),6)
            self.assertEqual(report['actual']['dense_trunk'],2)
            self.assertFalse(any('manifest' in p for p in report['shards']))

    def test_incomplete_refused_before_copy_and_corruption_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=self.fixture(Path(directory))
            report=runner.inventory(*roots,layers=2,counts={'spec1':1,'dense_trunk':2,'globals':1})
            self.assertFalse(report['complete'])
            self.assertTrue(any('L01' in m for m in report['missing']))
            with (roots[1]/'spec.safetensors').open('ab') as f:f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                runner.inventory(*roots,layers=1,counts={'spec1':1,'dense_trunk':2,'globals':1})

    def test_source_preflight_rejects_missing_coverage_before_payload_work(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory,patch('orca_export.index_shards',return_value={
            'coverage':{'missing':['tensor'],'unsupported_sources':[]},'missing_assets':[]}),\
             patch('orca_export.conversion_plan',return_value={}),\
             patch('check_full_candidate.committed_sources',return_value=({},{})):
            with self.assertRaisesRegex(ValueError,'coverage incomplete'):
                runner.preflight_sources(None,{'manifests':[],'shards':[]},{},Path(directory)/'preflight.json')

    def test_duplicate_or_partial_engram_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'Duplicate Engram'):
                runner.inventory(*roots[:3],[roots[3][0]]*2,layers=1,counts={'spec1':1,'dense_trunk':2,'globals':1})


if __name__=='__main__':unittest.main()
