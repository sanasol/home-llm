"""No GPU, network, or spawn required for driver planning/scheduling checks."""
from concurrent.futures import Future
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_orca_conversion as driver


class DriverTests(unittest.TestCase):
    def test_selection_policy_and_default(self):
        self.assertEqual(driver.selection('0,20,38-39',40),[0,20,38,39])
        jobs=driver.job_plan([0,20,39],[0,7,383],['gate','up','down'],policy={'20':{'7':{'down':40}}})
        self.assertEqual(len(jobs),27)
        self.assertEqual(next(j['n'] for j in jobs if j['id']=='L20-E007-down'),40)
        self.assertEqual(jobs[0]['n'],28)
        with self.assertRaises(ValueError):
            driver.job_plan([0],[0],['down'],policy={'0':{'0':{'down':True}}})
        with self.assertRaises(ValueError):
            driver.selection('40',40)

    def test_bounded_scheduler_propagates_results(self):
        class Pool:
            count=0
            def submit(self,call,job):
                self.count+=1
                future=Future()
                future.set_result({'id':job})
                return future
        pools=[Pool(),Pool()]
        results=[]
        driver.run_bounded(list(range(19)),pools,results.append,2)
        self.assertEqual(sorted(r['id'] for r in results),list(range(19)))
        self.assertEqual(sum(p.count for p in pools),19)

    def test_dry_run_never_imports_torch_or_starts_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,str(Path(driver.__file__)),directory,'--layers','0,20,39',
                                   '--experts','0,7,383','--gpus','0,1','--workers-per-gpu','2'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads((Path(directory)/'job-plan.json').read_text())
            self.assertEqual(len(plan['jobs']),27)
            self.assertEqual(plan['max_pending'],8)
            self.assertFalse((Path(directory)/'manifest.json').exists())

    def test_output_lock_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as directory:
            with driver.output_lock(directory):
                with self.assertRaisesRegex(ValueError,'already owned'):
                    driver.output_lock(directory)
            with driver.output_lock(directory):
                pass
            try:
                with driver.output_lock(directory):
                    raise RuntimeError('failure')
            except RuntimeError:
                pass
            with driver.output_lock(directory):
                pass

    def test_unvisited_policy_never_accepts_bad_observed_stats(self):
        job={'id':'L21-E207-down','n':28}
        settings={'unvisited_policy':'identity-n28','unvisited_jobs':[job['id']]}
        self.assertTrue(driver.allow_unvisited_fallback(job,settings,ValueError('Unvisited expert or invalid calibration count')))
        self.assertFalse(driver.allow_unvisited_fallback(job,settings,ValueError('all-zero importance')))
        self.assertFalse(driver.allow_unvisited_fallback(dict(job,n=24),settings,ValueError('Unvisited expert')))
        self.assertFalse(driver.allow_unvisited_fallback(dict(job,id='other'),settings,ValueError('Unvisited expert')))

    def test_crash_after_encode_repairs_fallback_metrics_on_resume(self):
        from types import SimpleNamespace
        import orca_export
        import numpy as np
        job=driver.job_plan([21],[207],['down'])[0]
        class Importance:
            def read_native(self,native):
                raise ValueError('Unvisited expert or invalid calibration count')
        reader=SimpleNamespace(tensors={'blk.21.ffn_down_exps.weight':SimpleNamespace(shape=(384,128,128))})
        with tempfile.TemporaryDirectory() as directory:
            settings={'output':directory,'seed':0,'batch':1,'policy_sha256':'policy','imatrix_identity':{},
                      'unvisited_policy':'identity-n28','unvisited_jobs':[job['id']],
                      'imatrix_reader_sha256':'code','coverage_sha256':'coverage'}
            driver._WORKER=(reader,Importance(),settings)
            def encoded(reader,manifest,layer,expert,projection,n,params,h):
                filename=f'expert.{expert}.{projection}.safetensors'
                orca_export.write_safetensors(manifest.root/filename,{'x':np.ones(1,np.float32)}, {})
                manifest.finish(f'expert.{expert}.{projection}',filename,{'calibrated':True})
                return manifest.root/filename
            original=orca_export.atomic_json
            def crash(path,state):
                if state.get('complete') and any(not r.get('metrics',{}).get('calibrated',True) for r in state['complete'].values()):
                    raise RuntimeError('crash after projection finish')
                return original(path,state)
            with patch('orca_export.source_pin',side_effect=lambda r,p:{'params':p}),patch('orca_export.projection',side_effect=encoded),patch('orca_export.atomic_json',side_effect=crash):
                with self.assertRaisesRegex(RuntimeError,'after projection'):
                    driver.run_job(job)
            with patch('orca_export.source_pin',side_effect=lambda r,p:{'params':p}),patch('orca_export.projection') as encode:
                result=driver.run_job(job)
                encode.assert_not_called()
                state=json.loads(Path(result['manifest']).read_text())
                self.assertFalse(state['complete']['expert.207.down']['metrics']['calibrated'])
                self.assertEqual(next(iter(state['importance_provenance'].values()))['status'],'unobserved-explicit-identity-fallback')

    def test_partial_layer_never_assembles(self):
        with self.assertRaisesRegex(ValueError,'all384'):
            driver.finalize_layer(0,driver.job_plan([0],[0],['down']),None,None,'unused')


if __name__=='__main__':
    unittest.main()
