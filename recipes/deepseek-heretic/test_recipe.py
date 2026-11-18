import importlib.util
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import numpy as np
import torch
import build
import stq
import hub
from assemble_root import verify_overlay


class RecipeTest(unittest.TestCase):
    def test_https_redirect_does_not_forward_hf_credentials_cross_origin(self):
        with patch.object(hub, 'TOKEN', 'synthetic-test-token'):
            request = hub._req('https://huggingface.co/repo/file', {'Range': 'bytes=4-7'})
            handler = hub.SafeRedirect()
            same = handler.redirect_request(request, None, 302, 'Found', {}, 'https://huggingface.co/another')
            other = handler.redirect_request(request, None, 302, 'Found', {}, 'https://cdn-lfs.hf.co/signed')
            self.assertEqual(same.get_header('Authorization'), 'Bearer synthetic-test-token')
            self.assertIsNone(other.get_header('Authorization'))
            self.assertEqual(other.get_header('Range'), 'bytes=4-7')
            self.assertEqual(other.get_method(), 'GET')
            self.assertIsNone(hub._req('https://example.com/file').get_header('Authorization'))
            for url in ['http://huggingface.co/file', 'https://user:password@example.com/file']:
                with self.assertRaises(ValueError):
                    handler.redirect_request(request, None, 302, 'Found', {}, url)
            with self.assertRaises(ValueError):
                hub.urlopen('http://huggingface.co/file')

    def test_base_index_atomic_parallel_fetch_and_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(build, 'WORK', directory), patch.object(build, '_cache', {}):
            target = Path(directory) / f'base_index.{build.BASE_REV}.json'
            started, release = threading.Event(), threading.Event()
            raw = json.dumps({'weight_map': {'tensor': 'shard.safetensors'}}).encode()
            def fetch(*args):
                started.set()
                self.assertTrue(release.wait(5))
                return raw
            with patch.object(build, 'rng', side_effect=fetch):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a = pool.submit(build.base_index)
                    self.assertTrue(started.wait(5))
                    self.assertFalse(target.exists())
                    b = pool.submit(build.base_index)
                    self.assertFalse(target.exists())
                    release.set()
                    self.assertEqual(a.result(), {'tensor': 'shard.safetensors'})
                    self.assertEqual(b.result(), a.result())
            self.assertEqual(json.loads(target.read_text())['weight_map'], {'tensor': 'shard.safetensors'})
            before = target.read_bytes()
            with patch.object(build, '_cache', {}), patch.object(build, 'rng', side_effect=OSError('network failed')):
                self.assertEqual(build.base_index(), {'tensor': 'shard.safetensors'})
            self.assertEqual(target.read_bytes(), before)
            target.write_bytes(b'old-truncated-json')
            with patch.object(build, '_cache', {}), patch.object(build, 'rng', side_effect=OSError('network failed')):
                with self.assertRaises(OSError): build.base_index()
            self.assertEqual(target.read_bytes(), b'old-truncated-json')
            with patch.object(build, '_cache', {}), patch.object(build, 'rng', return_value=raw):
                self.assertEqual(build.base_index(), {'tensor': 'shard.safetensors'})
            self.assertEqual(target.read_bytes(), raw)
        with tempfile.TemporaryDirectory() as directory, patch.object(build, 'WORK', directory), patch.object(build, '_cache', {}), patch.object(build, 'rng', side_effect=OSError('network failed')):
            with self.assertRaises(OSError): build.base_index()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_only_38_heretic_banks_and_pinned_fp8_reads(self):
        self.assertEqual(build.tensors_of('heretic'), [(i, 'wo_b') for i in range(2, 40)])
        self.assertEqual(set(build.DONORS), {'heretic'})
        seen = []
        def tensor(repo, shard, name, header, rev):
            seen.append(rev)
            return bytes(1024 if name.endswith('.weight') else 1), {'dtype': 'F8_E4M3' if name.endswith('.weight') else 'F8_E8M0', 'shape': [32, 32] if name.endswith('.weight') else [1, 1]}
        with patch.object(build, 'hdr', return_value=({}, 0, None)), patch.object(build, 'st_tensor', side_effect=tensor):
            build.fp8_raw(build.BASE, 'layers.2.attn.wo_b', 'model.safetensors')
        self.assertEqual(seen, [build.BASE_REV, build.BASE_REV])

    def test_fp8_scale_and_stored_scale_transform_inverse(self):
        codes = np.full((128, 128), 0x38, np.uint8)
        scales = np.full((4, 4), 127, np.uint8)
        np.testing.assert_array_equal(stq.fp8_dense(codes, scales), np.ones((128, 128)))
        scales[0, 0] = 255
        with self.assertRaises(ValueError): stq.fp8_dense(codes, scales)
        weight = torch.arange(128*128, dtype=torch.float32).reshape(128, 128) / 10000
        su = np.full(128, .5, np.float16); sv = np.full(128, 2, np.float16)
        inner = stq.target_inner(weight, su, sv)
        reconstructed = torch.from_numpy(su.astype(np.float32))[:, None] * stq.rotate(inner, stq.hadamard()) * torch.from_numpy(sv.astype(np.float32))[None, :]
        torch.testing.assert_close(reconstructed, weight, rtol=2e-5, atol=2e-5)

    def test_actual_small_spec1_search_and_public_decode(self):
        torch.set_num_threads(1)
        target = torch.zeros((1, 256), dtype=torch.float32)
        packed, error = stq.encode_tiles(target, 64, batch=1)
        self.assertEqual(packed.shape, (1, 64))
        self.assertEqual(packed.dtype, np.uint16)
        inner = stq.inner_from_trellis(packed.reshape(1, 1, 64))
        self.assertTrue(np.isfinite(inner).all())
        self.assertGreaterEqual(float(error[0]), 0)
        self.assertAlmostEqual(float(np.square(inner.astype(np.float64)).sum()), float(error[0]), places=7)

    def test_overlay_rejects_non_target_byte_changes(self):
        info = {'dtype': 'U16', 'shape': [512, 320, 64], 'data_offsets': [4, 4+512*320*64*2]}
        header = json.dumps({'layers.2.attn.wo_b.trellis': info}).encode()
        original = struct.pack('<Q', len(header)) + header + b'HEAD' + bytes(info['data_offsets'][1]-4) + b'TAIL'
        actual = bytearray(original); offset = 8+len(header)+4; actual[offset] = 1
        verify_overlay(bytes(actual), original, 2)
        actual[-1] ^= 1
        with self.assertRaises(ValueError): verify_overlay(bytes(actual), original, 2)


if __name__ == '__main__': unittest.main()
