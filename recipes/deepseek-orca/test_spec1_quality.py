"""CPU-only integration checks; synthetic codec is NOT a real SPEC1 quality test."""
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import spec1_quality as spec
import stq_quality as q


def fake_codec(tiles):
    # Reproducible scalar stand-in exposes feedback, grouping and stored-scale bugs.
    codes = np.zeros((len(tiles), 96), dtype=np.uint16)
    return codes, np.rint(tiles * 4) / 4


class Spec1QualityChecks(unittest.TestCase):
    def test_edited_donor_is_source_and_loss_uses_stored_scales(self):
        rng = np.random.default_rng(19)
        donor = rng.normal(size=(128, 128))
        original = donor.copy()
        h = np.eye(128)
        result = spec.encode_matrix(donor, h, encode=fake_codec, gains=(0.8, 1, 1.2))
        self.assertEqual(result['trellis'].shape, (8, 8, 96))
        self.assertEqual(result['suh'].dtype, np.float16)
        self.assertEqual(result['svh'].dtype, np.float16)
        inner = q.target_inner(donor, result['suh'], result['svh'])
        l, _, _ = q.prepare_ldl(q.rotated_hessian(h, result['suh']))
        _, decoded = q.ldlq(inner, l, fake_codec)
        loss = q.weighted_error(donor, q.reconstruct(decoded, result['suh'], result['svh']), h)
        self.assertEqual(loss, result['calibration_loss'])
        np.testing.assert_array_equal(donor, original)
        edited = donor.copy()
        edited[:, 3] += 8
        modified = spec.encode_matrix(edited, h, encode=fake_codec)
        self.assertNotEqual(result['donor_sha256'], modified['donor_sha256'])
        self.assertFalse(np.array_equal(result['svh'], modified['svh']))

    def test_grouped_wo_a_equals_independent_groups(self):
        rng = np.random.default_rng(20)
        donor = rng.normal(size=(2, 128, 128))
        h = np.stack([np.eye(128), 3 * np.eye(128)])
        result = spec.encode_trunk(donor, h, name='layers.2.attn.wo_a', encode=fake_codec)
        self.assertEqual(result['trellis'].shape, (2, 8, 8, 96))
        for group in range(2):
            single = spec.encode_matrix(donor[group], h[group],
                                       name=f'layers.2.attn.wo_a.group{group}', encode=fake_codec)
            for key in ('trellis', 'suh', 'svh'):
                np.testing.assert_array_equal(result[key][group], single[key])
        diagonal = spec.encode_trunk(donor, np.diagonal(h, axis1=1, axis2=2),
                                     name='layers.2.attn.wo_a', encode=fake_codec)
        for key in ('trellis', 'suh', 'svh'):
            np.testing.assert_array_equal(diagonal[key], result[key])
        self.assertAlmostEqual(diagonal['calibration_loss'], result['calibration_loss'])
        with self.assertRaises(ValueError):
            spec.encode_trunk(donor, h[0], encode=fake_codec)

    def test_stq_adapter_permutes_once_and_decodes_actual_bytes(self):
        perm = np.arange(256)[::-1]
        tiles = np.arange(2 * 256).reshape(2, 16, 16).astype(float)
        packed = np.zeros((2, 96), dtype=np.uint16)
        values = np.arange(256)[None, :].repeat(2, axis=0)
        observed = []

        class Array:
            def __init__(self, x):
                self.x = x
            def to(self, device):
                self.device = device
                return self

        def search(x, n, batch):
            observed.append(x)
            self.assertEqual((n, batch, x.device), (96, 2, 'cpu'))
            return packed, np.zeros(2)

        stq = types.SimpleNamespace(PERM=perm, W_BITS=15, encode_tiles=search,
                                    spec1_table=lambda n: np.arange(32768),
                                    unpack_codes=lambda codes: values)
        torch = types.SimpleNamespace(from_numpy=Array)
        with patch.dict(sys.modules, {'stq': stq, 'torch': torch}):
            codes, decoded = spec.stq_encoder(96, batch=2)(tiles)
        np.testing.assert_array_equal(observed[0].x, tiles.reshape(-1, 256)[:, perm])
        np.testing.assert_array_equal(decoded.reshape(-1, 256)[:, perm], values)
        np.testing.assert_array_equal(codes, packed)

    def test_real_stq_cpu_tile_when_torch_is_installed(self):
        try:
            import torch
            import stq
        except ModuleNotFoundError:
            self.skipTest('existing stq backend requires torch')
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            tiles = np.random.default_rng(21).normal(size=(1, 16, 16))
            codes, decoded = spec.stq_encoder(96)(tiles)
            expected = stq.inner_from_trellis(codes.reshape(1, 1, 96))
            np.testing.assert_array_equal(decoded[0], expected)
            self.assertTrue(np.isfinite(decoded).all())
        finally:
            torch.set_num_threads(previous_threads)

    def test_tied_rates_bound_group_count_and_exact_average(self):
        items = [{'weights': 3 * 256, 'candidates': [
            {'n': 24, 'loss': loss}, {'n': 32, 'loss': 0}]} for loss in [100, 1]]
        result = spec.allocate_tied_experts(items)
        self.assertEqual(result['rates_n'], [32, 24])
        self.assertEqual(result['average_bpw'], 1.75)
        self.assertEqual(result['rate_plan']['0'], {'gate': 32, 'up': 32, 'down': 32})
        self.assertLessEqual(len(result['group_signatures']), 32)
        with self.assertRaises(ValueError):
            spec.allocate_tied_experts([{'weights': 256, 'candidates': [{'n': 16, 'loss': 1}]}])

    def test_invalid_rates_gains_scales_and_codec_fail(self):
        for n in [28, 128, 0]:
            with self.assertRaises(ValueError):
                spec.stq_encoder(n)
        with self.assertRaises(ValueError):
            spec.encode_matrix(np.ones((128, 128)), np.eye(128), gains=(0,), encode=fake_codec)
        with self.assertRaises(ValueError):
            spec._stored_scale([1e-30])
        with self.assertRaises(ValueError):
            spec.encode_matrix(np.ones((128, 128)), np.eye(128), n=64, encode=fake_codec)


if __name__ == '__main__':
    unittest.main()
