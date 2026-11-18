"""CPU-only adapter checks; live bounded sample is an explicit separate CLI run."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import engram_export as export
from orca_reader import Tensor
from pair1_contract import read_header


class FakeReader:
    def __init__(self, fail_at=None):
        self.tensors = {export.DONOR: Tensor("fake.gguf", (256, 768022850), "IQ4_NL", 0, 32)}
        self.calls = []
        self.fail_at = fail_at

    def read(self, name, row_start, row_stop):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("simulated interrupted source")
        self.calls.append((row_start, row_stop))
        return (np.arange((row_stop-row_start)*256, dtype=np.float32).reshape(-1,256) % 33 - 16) / 9


class EngramTests(unittest.TestCase):
    def test_local_source_cli_passthrough(self):
        import sys
        with tempfile.TemporaryDirectory() as directory, patch.object(sys,'argv',['engram_export.py',str(Path(directory)/'output.safetensors'),'--layer','1',
                                     '--catalog','catalog.json','--local-source-dir','/cache/root','--row-count','4']), \
             patch('orca_reader.OrcaReader') as reader, \
             patch('engram_export.export_table',return_value={'done':True}) as convert, \
             patch('builtins.print'):
            export.main()
        reader.assert_called_once_with('catalog.json',local_source_dir='/cache/root')
        self.assertEqual(convert.call_args.kwargs['row_count'],4)

    def test_cuda_dispatch_keeps_gpu_metrics_without_cpu_reconstruction(self):
        def quantize(x,metrics):
            q,s=export.quantize_rows(x)
            return q,s,{'sse':2.,'signal_sq':8.}
        with tempfile.TemporaryDirectory() as directory,patch('engram_cuda.quantize_rows',side_effect=quantize) as cuda,patch('engram_export.dequantize_rows',side_effect=AssertionError('CPU metrics not allowed')):
            state=export.export_table(FakeReader(),Path(directory)/'cuda.safetensors',1,row_count=2,backend='cuda')
            self.assertEqual(state['sse'],2.)
            self.assertEqual(state['relative_frobenius'],.5)
            self.assertEqual(state['pin']['backend'],'cuda')
            cuda.assert_called_once()

    def test_fp8_torch_oracle(self):
        rng = np.random.default_rng(42)
        x = (rng.standard_normal((100,256)) * rng.lognormal(0, 8, (100,8)).repeat(32,axis=1)).astype(np.float32)
        x[0] = 0
        x[1] = -0.0
        q, s = export.quantize_rows(x)
        scales = torch.from_numpy(np.repeat(np.exp2(s.astype(np.float32)-127),32,axis=1))
        expected = (torch.from_numpy(x) / scales).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        np.testing.assert_array_equal(q, expected)
        decoded = torch.from_numpy(q).view(torch.float8_e4m3fn).float() * scales
        np.testing.assert_array_equal(export.dequantize_rows(q,s), decoded.numpy())
        self.assertFalse(np.any(s == 255))

    def test_round_nearest_even_boundaries(self):
        mids = (export.VALUES[:-1] + export.VALUES[1:]) / 2
        # A max of448 enforces scale1; all exact representable midpoints exercise ties.
        x = np.zeros((len(mids),32),np.float32)
        x[:,0], x[:,1] = mids, 448
        q, s = export.quantize_rows(x)
        expected = np.arange(len(mids)) + (np.arange(len(mids)) & 1)
        np.testing.assert_array_equal(q[:,0],expected)
        np.testing.assert_array_equal(s,127)
        with self.assertRaises(ValueError):
            export.quantize_rows(np.full((1,32),np.nan,np.float32))

    def test_recovers_crash_after_final_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.safetensors'
            original = export.atomic_json
            def crash(path, state):
                if state.get('done'):
                    raise RuntimeError('crash after rename')
                return original(path, state)
            with patch('engram_export.atomic_json',side_effect=crash):
                with self.assertRaisesRegex(RuntimeError,'after rename'):
                    export.export_table(FakeReader(),path,1,row_count=2,chunk_rows=2)
            reader = FakeReader()
            state = export.export_table(reader,path,1,row_count=2,chunk_rows=2)
            self.assertTrue(state['done'])
            self.assertEqual(reader.calls,[])

    def test_stream_resume_unequal_layer_offset_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.safetensors"
            reader = FakeReader(fail_at=1)
            with self.assertRaisesRegex(RuntimeError,"interrupted"):
                export.export_table(reader,path,14,row_start=3,row_count=7,chunk_rows=3)
            reader.fail_at = None
            state = export.export_table(reader,path,14,row_start=3,row_count=7,chunk_rows=3)
            self.assertEqual(reader.calls,[(384006171,384006174),(384006174,384006177),(384006177,384006178)])
            self.assertTrue(state['done'])
            h = read_header(path)
            self.assertEqual(h['layers.14.engram.embed.weight']['shape'],[7,256])
            self.assertEqual(h['layers.14.engram.embed.scale']['shape'],[7,8])
            self.assertIn('sample',h['__metadata__']['status'])
            self.assertEqual(state,export.export_table(reader,path,14,row_start=3,row_count=7,chunk_rows=3))
            with self.assertRaisesRegex(ValueError,'pin mismatch'):
                export.export_table(reader,path,14,row_start=3,row_count=7,chunk_rows=2)
            with path.open('r+b') as f:
                f.seek(-1,2)
                f.write(b'\xff')
            with self.assertRaisesRegex(ValueError,'changed'):
                export.export_table(reader,path,14,row_start=3,row_count=7,chunk_rows=3)


if __name__ == '__main__':
    unittest.main()
