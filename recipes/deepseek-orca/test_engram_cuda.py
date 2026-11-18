"""CPU torch oracle by default; CUDA parity only when explicitly requested."""
import os
import unittest

import numpy as np
import engram_cuda
from engram_export import quantize_rows,dequantize_rows,VALUES


class CudaAdapterTests(unittest.TestCase):
    def check(self,device):
        rng=np.random.default_rng(7)
        x=(rng.normal(size=(512,256))*rng.lognormal(0,8,(512,8)).repeat(32,axis=1)).astype(np.float32)
        x[0]=0
        x[1]=-0.
        mids=(VALUES[:-1]+VALUES[1:])/2
        x[2:128,0]=mids
        x[2:128,1]=448
        expected_q,expected_s=quantize_rows(x)
        q,s,metrics=engram_cuda.quantize_rows(x,device=device,metrics=True)
        np.testing.assert_array_equal(q,expected_q)
        np.testing.assert_array_equal(s,expected_s)
        expected_error=x.astype(np.float64)-dequantize_rows(q,s).astype(np.float64)
        self.assertAlmostEqual(metrics['sse']/float(np.square(expected_error).sum()),1.,places=13)
        self.assertAlmostEqual(metrics['signal_sq']/float(np.square(x.astype(np.float64)).sum()),1.,places=13)

    def test_cpu_torch_exact_codes_scales_metrics(self):
        self.check('cpu')

    @unittest.skipUnless(os.environ.get('ENGRAM_CUDA_TEST')=='1','CUDA requires explicit lead authorization')
    def test_cuda_exact_codes_scales_metrics(self):
        self.check('cuda')

    def test_invalid_input_fails_without_cuda(self):
        with self.assertRaises(ValueError):
            engram_cuda.quantize_rows(np.full((1,32),np.nan,np.float32),device='cpu')


if __name__=='__main__':
    unittest.main()
