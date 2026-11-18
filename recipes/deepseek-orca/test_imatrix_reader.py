"""CPU synthetic checks plus opt-in bounded actual donor calibration reads."""
import hashlib
import os
import struct
import unittest

import numpy as np
from imatrix_reader import ImatrixReader, donor_name, REPO, REVISION


def fixture(width=3, count=2.0, value=4.0):
    tensors = {
        "blk.20.ffn_down_exps.weight.in_sum2": ([width, 384], np.full((384, width), value, "<f4")),
        "blk.20.ffn_down_exps.weight.counts": ([1, 384], np.full(384, count, "<f4")),
        "blk.20.attn_output_a.weight.in_sum2": ([32768], np.full(32768, value, "<f4")),
        "blk.20.attn_output_a.weight.counts": ([1], np.array([10], "<f4")),
    }
    def string(s):
        return struct.pack("<Q", len(s)) + s.encode()
    meta = string("general.type") + struct.pack("<I", 8) + string("imatrix")
    for key, value in (("imatrix.chunk_count", 2), ("imatrix.chunk_size", 5)):
        meta += string(key) + struct.pack("<II", 4, value)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 3) + meta
    data = b""
    for name, (dims, array) in tensors.items():
        data = data.ljust((len(data) + 31) // 32 * 32, b"\0")
        header += string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", 0, len(data))
        data += array.tobytes()
    end = len(header)
    blob = (header.ljust((end + 31) // 32 * 32, b"\0") + data).ljust(262144, b"\0")
    calls = []
    def fetch(repo, path, start, stop, revision):
        assert (repo, path, revision) == (REPO, "imatrix.gguf", REVISION)
        calls.append((start, stop))
        return blob[start:stop + 1]
    return ImatrixReader(fetch=fetch), calls, hashlib.sha256(header).hexdigest()


class SyntheticTest(unittest.TestCase):
    def test_lazy_expert_frequency_and_provenance(self):
        reader, calls, header_hash = fixture()
        self.assertFalse(calls)
        self.assertEqual(reader.identity()["header_sha256"], header_hash)
        self.assertEqual(calls, [(0, 262143)])
        r = reader.read_native("layers.20.ffn.experts.7.w2.weight")
        np.testing.assert_array_equal(r.raw_sum2, [4, 4, 4])
        np.testing.assert_allclose(r.weights("layer"), [.4, .4, .4])
        np.testing.assert_array_equal(r.weights("expert"), [2, 2, 2])
        self.assertEqual(r.count, 2)
        self.assertEqual(r.provenance["expert"], 7)
        self.assertEqual(len(r.provenance["sample_sha256"]), 64)
        reader.read_native("layers.20.ffn.experts.383.w2.weight")
        self.assertEqual([b - a + 1 for a, b in calls], [262144, 1536, 12, 12])
        with self.assertRaises(ValueError):
            r.weights("unknown")

    def test_grouped_attention_requires_group(self):
        reader, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "Grouped"):
            reader.read_native("layers.20.attn.wo_a.weight")
        self.assertEqual(reader.read_native("layers.20.attn.wo_a.weight", group=7).raw_sum2.shape, (4096,))
        with self.assertRaises(ValueError):
            reader.read_native("layers.20.attn.wo_a.weight", group=8)

    def test_invalid_diagonals_and_counts_fail_closed(self):
        for count, value in [(0, 4), (-1, 4), (11, 4), (float("nan"), 4), (2, 0), (2, -1), (2, float("inf"))]:
            with self.subTest(count=count, value=value):
                reader, _, _ = fixture(count=count, value=value)
                with self.assertRaises(ValueError):
                    reader.read_native("layers.20.ffn.experts.7.w2.weight")
        with self.assertRaises(ValueError):
            donor_name("layers.40.ffn.experts.0.w2.weight")
        with self.assertRaises(ValueError):
            donor_name("layers.20.ffn.experts.384.w2.weight")
        with self.assertRaises(KeyError):
            donor_name("vision.blocks.0.mlp.w1.weight")

    def test_individual_inactive_channels_are_valid(self):
        reader, _, _ = fixture()
        original = reader._range
        def channel_zero(key, first, size):
            raw = original(key, first, size)
            return b"\0\0\0\0" + raw[4:] if key.endswith(".in_sum2") else raw
        reader._range = channel_zero
        result = reader.read_native("layers.20.ffn.experts.7.w2.weight")
        np.testing.assert_array_equal(result.raw_sum2, [0, 4, 4])

    def test_counts_only_coverage(self):
        reader, calls, _ = fixture()
        report = reader.coverage()
        self.assertEqual(report["projection_vectors"], 120)
        self.assertEqual(len(report["unverified_vectors"]), 119)
        self.assertEqual(report["zero_projection_experts"], 0)
        self.assertEqual([b - a + 1 for a, b in calls], [262144, 1536])

    def test_full_production_expert_mapping(self):
        for layer in range(40):
            for expert in range(384):
                for w, proj in [(1, "gate"), (2, "down"), (3, "up")]:
                    self.assertEqual(donor_name(f"layers.{layer}.ffn.experts.{expert}.w{w}.weight"),
                                     (f"blk.{layer}.ffn_{proj}_exps.weight", expert))


@unittest.skipUnless(os.environ.get("ORCA_LIVE_TEST") == "1", "bounded live HF opt-in")
class LiveTest(unittest.TestCase):
    def test_actual_l20_e7_diagonals(self):
        reader = ImatrixReader()
        self.assertEqual(reader.identity()["layer_count"], 614400)
        r = reader.read_native("layers.20.ffn.experts.7.w2.weight")
        self.assertEqual(r.count, 2364)
        self.assertEqual(r.raw_sum2.shape, (2304,))
        self.assertAlmostEqual(float(r.raw_sum2.mean()), 128.587, places=2)
        np.testing.assert_allclose(r.weights("layer") * 614400, r.raw_sum2)
        gate = reader.read_native("layers.20.ffn.experts.7.w1.weight")
        self.assertEqual(gate.count, r.count)
        self.assertEqual(gate.raw_sum2.shape, (5120,))
        self.assertEqual(reader.read_native("layers.20.attn.wo_b.weight").count, 614400)
        self.assertEqual(reader.read_native("layers.20.attn.wo_a.weight", group=7).raw_sum2.shape, (4096,))
        for layer in [0, 39]:
            # Actual first/last layer expert slices, not only names or synthetic offsets.
            for expert in [0, 383]:
                try:
                    edge = reader.read_native(f"layers.{layer}.ffn.experts.{expert}.w2.weight")
                except ValueError as e:
                    self.assertIn("Unvisited expert", str(e))
                else:
                    self.assertEqual(edge.raw_sum2.shape, (2304,))


if __name__ == "__main__":
    unittest.main()
