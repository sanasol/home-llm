"""Small CPU checks: python3 -m unittest discover -s ds-sushi-abl -p test_stq_quality.py."""
import itertools
import unittest
import numpy as np
import stq_quality as q


class QualityChecks(unittest.TestCase):
    def test_scales_and_hadamard_match_public_equation(self):
        rng = np.random.default_rng(7)
        w = rng.normal(size=(128, 256))
        signs = [rng.choice([-1, 1], size=n) for n in w.shape]
        inner, su, sv = q.regularize(w, *signs)
        np.testing.assert_allclose(q.reconstruct(inner, su, sv), w, atol=1e-12)
        # Store scales as the actual pack will, not imaginary f64 scales.
        su, sv = su.astype(np.float16), sv.astype(np.float16)
        inner = q.target_inner(w, su, sv)
        np.testing.assert_allclose(q.reconstruct(inner, su, sv), w, atol=1e-12)
        np.testing.assert_allclose(q.hadamard_axis(q.hadamard_axis(w, 0), 0), w, atol=1e-12)
        # Independent Sylvester construction, also used by stq.hadamard().
        h = np.ones((1, 1))
        for _ in range(7):
            h = np.block([[h, h], [h, -h]])
        np.testing.assert_allclose((h / np.sqrt(128)) @ w,
                                   q.hadamard_axis(w, 0), atol=1e-12)

    def test_weighted_error_and_rotated_hessian_include_output_scales(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(23, 128))
        h = x.T @ x
        w = rng.normal(size=(128, 128))
        _, su, sv = q.regularize(w, np.ones(128), np.ones(128))
        e = rng.normal(size=w.shape)
        public_e = q.reconstruct(e, su, sv)
        np.testing.assert_allclose(q.weighted_error(public_e, np.zeros_like(e), h),
                                   np.sum((x @ public_e) ** 2), rtol=1e-12)
        # Output H128 and svh do not disappear unless output scales are uniform.
        right_e = q.hadamard_axis(e, 1) * sv[None, :]
        np.testing.assert_allclose(np.sum(right_e * (q.rotated_hessian(h, su) @ right_e)),
                                   q.weighted_error(public_e, np.zeros_like(e), h), rtol=1e-12)

    def test_ldl_factor_and_reverse_feedback_geometry(self):
        h = np.block([[np.eye(16), np.eye(16) * 0.9],
                      [np.eye(16) * 0.9, np.eye(16)]])
        l, d, damped = q.prepare_ldl(h)
        a = np.eye(32) + l
        np.testing.assert_allclose(a @ d @ a.T, damped, atol=1e-12)
        seen = []

        def encode(tiles):
            seen.append(tiles.copy())
            return np.zeros((len(tiles), 28), dtype=np.uint16), np.rint(tiles)

        inner = np.full((32, 32), 0.49)
        packed, decoded = q.ldlq(inner, l, encode)
        self.assertEqual(packed.shape, (2, 2, 28))
        np.testing.assert_allclose(seen[0], 0.49)
        np.testing.assert_allclose(seen[1], 0.49 + (0.9 / 1.025) * 0.49)
        self.assertLess(q.weighted_error(inner, decoded, h),
                        q.weighted_error(inner, np.rint(inner), h))
        empty_l, _, _ = q.prepare_ldl(np.zeros((16, 16)))
        np.testing.assert_array_equal(empty_l, 0)
        rng = np.random.default_rng(13)
        x = rng.normal(size=(70, 48))
        l, d, damped = q.prepare_ldl(x.T @ x)
        a = np.eye(48) + l
        np.testing.assert_allclose(a @ d @ a.T, damped, atol=1e-12)
        unique = np.arange(32 * 32).reshape(32, 32).astype(float)

        def identity(tiles):
            codes = np.zeros((len(tiles), 28), dtype=np.uint16)
            codes[:, 0] = tiles[:, 0, 0]
            return codes, tiles

        codes, decoded = q.ldlq(unique, np.zeros((32, 32)), identity)
        np.testing.assert_array_equal(decoded, unique)
        np.testing.assert_array_equal(codes[:, :, 0], [[0, 16], [512, 528]])

    def test_budget_allocator_matches_exhaustive_unequal_sizes(self):
        items = [{"weights": weights, "candidates": [
            {"n": 24, "loss": base}, {"n": 28, "loss": base / 2},
            {"n": 32, "loss": base / 5}, {"n": 40, "loss": base / 8}]}
            for weights, base in [(256, 10), (512, 3), (256, 8)]]
        got = q.allocate_bits(items)
        possibilities = []
        for choices in itertools.product(*(i['candidates'] for i in items)):
            bits = sum(i['weights'] * c['n'] // 16 for i, c in zip(items, choices))
            if bits <= got['budget_bits']:
                possibilities.append((sum(c['loss'] for c in choices), bits))
        self.assertEqual((got['loss'], got['payload_bits']), min(possibilities))
        self.assertLessEqual(got['average_bpw'], 1.75)
        # Same average, different rates: two equal-size tensors at K1.5/K2.
        mixed = q.allocate_bits([{'weights': 256, 'candidates': [
            {'n': 24, 'loss': loss}, {'n': 32, 'loss': 0}]} for loss in [100, 1]])
        self.assertEqual(mixed['rates_n'], [32, 24])
        self.assertEqual(mixed['average_bpw'], 1.75)

    def test_native_diagonal_blocks_match_full_dense_path(self):
        rng = np.random.default_rng(25)
        # Unequal block scales catch accidentally local damping means; zero
        # channels and an all-zero block catch the empty-diagonal convention.
        diagonal = rng.uniform(0.01, 3, 384)
        diagonal[128:256] = 0
        diagonal[::17] = 0
        su = rng.uniform(-2, 2, 384)
        su[256:] *= 10
        full_h = q.rotated_hessian(np.diag(diagonal), su)
        l, _, _ = q.prepare_ldl(full_h)
        blocks = q.prepare_diagonal_ldl(diagonal, su)
        dense_blocks = np.zeros_like(l)
        for g, block in enumerate(blocks):
            dense_blocks[g * 128:(g + 1) * 128, g * 128:(g + 1) * 128] = block
        np.testing.assert_allclose(dense_blocks, l, rtol=1e-12, atol=1e-12)
        inner = rng.normal(size=(384, 32))
        def codec(tiles):
            return np.zeros((len(tiles), 28), dtype=np.uint16), np.rint(tiles * 3) / 3
        packed, decoded = q.ldlq(inner, l, codec)
        block_packed, block_decoded = q.ldlq_blocks(inner, blocks, codec)
        np.testing.assert_array_equal(block_packed, packed)
        np.testing.assert_array_equal(block_decoded, decoded)
        for values in (inner, np.full_like(inner, 0.49)):
            def distinct_codec(tiles):
                rec = np.rint(tiles * 3) / 3
                codes = np.zeros((len(tiles), 28), dtype=np.uint16)
                codes[:, 0] = (np.rint(rec[:, 0, 0] * 3).astype(np.int64) % 65536).astype(np.uint16)
                return codes, rec
            expected_codes, expected_rec = q.ldlq_blocks(values, blocks, distinct_codec)
            calls = []
            def batched_codec(tiles):
                calls.append(len(tiles))
                return distinct_codec(tiles)
            parallel_codes, parallel_rec = q.ldlq_blocks_parallel(values, blocks, batched_codec)
            np.testing.assert_array_equal(parallel_codes, expected_codes)
            np.testing.assert_array_equal(parallel_rec, expected_rec)
            self.assertEqual(calls, [6] * 8)
        np.testing.assert_allclose(q.weighted_error_diagonal(inner, decoded, diagonal),
                                   q.weighted_error(inner, decoded, np.diag(diagonal)), rtol=1e-12)
        with self.assertRaises(ValueError):
            q.prepare_diagonal_ldl(np.full(128, -1), np.ones(128))
        with self.assertRaises(ValueError):
            q.ldlq_blocks(inner, blocks[:1], codec)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            q.allocate_bits([{'weights': 256, 'candidates': [{'n': 32, 'loss': 1}]}])
        with self.assertRaises(ValueError):
            q.prepare_ldl(np.diag([-1.] + [1.] * 15))
        with self.assertRaises(ValueError):
            q.target_inner(np.ones((128, 128)), np.zeros(128), np.ones(128))
        with self.assertRaises(ValueError):
            q.ldlq(np.ones((16, 16)), np.eye(16), lambda x: None)
        with self.assertRaises(ValueError):
            q.weighted_error(np.ones((16, 16)), np.zeros((16, 16)), np.full((16, 16), np.nan))


if __name__ == '__main__':
    unittest.main()
