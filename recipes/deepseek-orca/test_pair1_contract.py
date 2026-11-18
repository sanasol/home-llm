"""CPU-only independent checks; SUSHI_SRC selects an existing public checkout."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pair1_contract as c


class ContractTests(unittest.TestCase):
    def test_frozen_table_and_public_oracle(self):
        self.assertEqual(c.table_digest(), "c089b4eb")
        path = Path(os.environ.get("SUSHI_SRC", "/tmp/sushi-src")) / "src/exl3/fixtures/exl3_pair1_tiles.safetensors"
        self.assertTrue(path.is_file(), "Set SUSHI_SRC to public Sushi checkout; do not replace oracle with encoder output")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "bfb3428e0657d0141d1a3c2cae254d63dfab8ae4d3c7a3ffd013c76ac7d8ddbe")
        self.assertEqual(c.read_header(path)["__metadata__"]["sashimi"],
                         "01b215e0337a71ea0f32f41567e234c4816ae1cb")
        fixture = c.read_tensors(path)
        for n in range(24, 65, 2):
            with self.subTest(n=n):
                actual = c.decode_tiles(fixture[f"trellis_n{n}"]).view(np.uint16)
                np.testing.assert_array_equal(actual, fixture[f"decoded_n{n}"])
                for chunk in (1, 2):
                    np.testing.assert_array_equal(c.decode_tiles(fixture[f"trellis_n{n}"], chunk).view(np.uint16), actual)

    def test_decode_chunk_bounds_and_equivalence(self):
        packed = np.random.default_rng(5).integers(0, 65536, (3, 173, 28), dtype=np.uint16)
        expected = c.decode_tiles(packed, chunk_tiles=1000)
        original = c.unpack_codes
        with patch.object(c, "unpack_codes", wraps=original) as unpack:
            actual = c.decode_tiles(packed)
        self.assertEqual([call.args[0].shape[0] for call in unpack.call_args_list], [512, 7])
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        np.testing.assert_array_equal(c.decode_tiles(packed[:, ::2], 17), expected[:, ::2])
        with self.assertRaises(ValueError):
            c.decode_tiles(packed, chunk_tiles=0)

    def test_pair_major_and_scale_orientation(self):
        rng = np.random.default_rng(34)
        tr = rng.integers(0, 65536, size=(2, 8, 16, 26), dtype=np.uint16)
        pm = tr.reshape(2, 8, 8, 2, 26).swapaxes(1, 2)
        np.testing.assert_array_equal(c.inner_from_trellis(tr), c.inner_from_trellis(pm, "pair_major"))
        su, sv = np.ones((2, 128), np.float16), np.ones((2, 256), np.float16)
        expected = c.reconstruct(tr, su, sv)
        np.testing.assert_array_equal(expected, c.reconstruct(pm, su, sv, "pair_major"))
        doubled = c.reconstruct(tr, su * np.float16(2), sv)
        np.testing.assert_array_equal(doubled, expected * 2)

    @staticmethod
    def layer():
        tensors = {}
        for g, e in enumerate((2, 3)):
            for p, i, o, n in (("gate", 128, 256, 24), ("up", 128, 256, 32), ("down", 256, 128, 26)):
                base = f"model.layers.0.mlp.switch_mlp.{p}_proj.g{g}"
                for part, dtype, shape in (("trellis", "U16", [e, i // 16, o // 16, n]), ("suh", "F16", [e, i]), ("svh", "F16", [e, o])):
                    tensors[base + "." + part] = dict(dtype=dtype, shape=shape)
        return tensors

    def test_groups_router_and_bad_metadata(self):
        tensors, quant = self.layer(), c.PAIR1_CONFIG.copy()
        self.assertEqual([x["experts"] for x in c.validate_routed(tensors, quant, 128, 256, 5, 3)], [2, 3])
        for mutate in (lambda x: x.pop(next(iter(x))),
                       lambda x: x[next(iter(x))].update(dtype="I16"),
                       lambda x: x[next(iter(x))].update(shape=[2, 8, 16, 22])):
            bad = copy.deepcopy(tensors)
            mutate(bad)
            with self.assertRaises(ValueError):
                c.validate_routed(bad, quant, 128, 256, 5, 3)
        with self.assertRaises(ValueError):
            c.validate_routed(tensors, quant, 128, 256, 6, 3)
        for key, value in (("window", 15), ("window", True), ("k", 1), ("k", float("nan")), ("codebook_digest", "deadbeef"), ("tile_order", "pack")):
            with self.assertRaises(ValueError):
                c.validate_quant(dict(quant, **{key: value}))
        c.validate_permutation([3, 0, 4, 1, 2], 5)
        with self.assertRaises(ValueError):
            c.validate_permutation([3, 0, 4, 1, 1], 5)

    def test_pair_major_header_and_stamp(self):
        tensors = self.layer()
        pm = {}
        for name, info in tensors.items():
            info = copy.deepcopy(info)
            if name.endswith(".trellis"):
                e, it, ot, n = info["shape"]
                info["shape"] = [e, ot // 2, it, 2, n]
                name += "_pm"
            pm[name] = info
        quant = dict(c.PAIR1_CONFIG, tile_order="pair_major")
        c.validate_routed(pm, quant, 128, 256, 5, 3)
        stamp = dict(format="stq", codebook="pair1", window="16", k="2", tile_order="pair_major", pack_revision="orca-pilot")
        c.validate_stamp(stamp, quant, "orca-pilot")
        with self.assertRaises(ValueError):
            c.validate_stamp(dict(stamp, tile_order=""), quant)
        with self.assertRaises(ValueError):
            c.validate_stamp(stamp, quant, "base-revision")

    def test_trunk_and_safetensors_bounds(self):
        quant = dict(format="stq", codebook="spec1", codebook_table="spec1-w15", codebook_digest="37f72431", k=8, window=15)
        tensors = {"head.trellis": dict(dtype="U16", shape=[8, 16, 128]),
                   "head.suh": dict(dtype="F16", shape=[128]), "head.svh": dict(dtype="F16", shape=[256])}
        self.assertEqual(c.validate_trunk(tensors, quant), 1)
        with self.assertRaises(ValueError):
            c.validate_trunk(dict(tensors, **{"head.weight": dict(dtype="BF16", shape=[256, 128])}), quant)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.safetensors"
            header = {"x": dict(dtype="U16", shape=[2], data_offsets=[0, 4])}
            raw = json.dumps(header).encode()
            path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 4)
            self.assertEqual(c.read_header(path), header)
            with path.open("ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                c.read_header(path)


if __name__ == "__main__":
    unittest.main()
