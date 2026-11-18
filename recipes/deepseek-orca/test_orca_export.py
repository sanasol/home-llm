"""CPU-only integration checks; no network, weights, or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import orca_export as export
import pair1_contract as contract


class ExportTests(unittest.TestCase):
    def test_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            pin = {"revision": "abc", "n": 28}
            m = export.Manifest(directory, pin)
            path = Path(directory) / "one.safetensors"
            export.write_safetensors(path, {"w": np.ones((2, 3), np.float32)}, {"pilot": "true"})
            m.finish("one", path.name)
            self.assertTrue(export.Manifest(directory, pin).ready("one"))
            with self.assertRaisesRegex(ValueError, "pin mismatch"):
                export.Manifest(directory, dict(pin, n=32))
            with path.open("ab") as f:
                f.write(b"bad")
            with self.assertRaisesRegex(ValueError, "changed"):
                m.ready("one")

    def test_fused_gate_roundtrip_is_f32_not_original_inverse(self):
        bits = np.array([0, 0x80000000, 0x3F800001, 0xBF7FFFFF, 0x00800001], np.uint32)
        gate = bits.view(np.float32).reshape(1, -1)
        q, k = export.fused_engram(gate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.safetensors"
            export.write_safetensors(path, {"q": q, "k": k}, {"representation": "fused-times-one"})
            stored = contract.read_tensors(path)
            product = stored["q"].astype(np.float32) * stored["k"].astype(np.float32)
            np.testing.assert_array_equal(product.view(np.uint32), bits.reshape(1, -1))
        with self.assertRaises(ValueError):
            export.fused_engram(gate.astype(np.float16))

    def test_plan_and_partial_global_index(self):
        from orca_reader import Tensor
        class Reader:
            tensors = {"output_norm.weight": Tensor("x", (128,), "F32", 0, 0),
                       "unmapped.weight": Tensor("x", (128,), "F32", 512, 0)}
        plan = export.conversion_plan(Reader())
        self.assertEqual(plan['donor_tensors'],2)
        self.assertEqual(plan['mappings'][1]['status'],'unsupported')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'norm.safetensors'
            export.write_safetensors(path,{'norm.weight':np.ones(128,np.float32)}, {})
            result = export.index_shards(plan,[path],Path(directory)/'index.json')
            self.assertEqual(result['coverage']['missing'],[])
            self.assertEqual(result['coverage']['unsupported_sources'],['unmapped.weight'])
            self.assertIn('config.json',result['missing_assets'])
            self.assertNotEqual(result['metadata']['status'],'complete')

    def test_materialized_root_discovery_and_bf16(self):
        import torch
        from orca_reader import Tensor
        class Reader:
            tensors={'output_norm.weight':Tensor('x',(128,),'F32',0,0)}
        values=np.array([0.,-0.,1.,1.00390625,1.01171875,-2.25],np.float32)
        np.testing.assert_array_equal(export.bf16_bits(values),torch.from_numpy(values).to(torch.bfloat16).view(torch.uint16).numpy())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'input.safetensors'
            export.write_safetensors(source,{'norm.weight':export.bf16_bits(values)}, {}, {'norm.weight':'BF16'})
            self.assertEqual(contract.read_header(source)['norm.weight']['dtype'],'BF16')
            plan=export.conversion_plan(Reader())
            config=root/'config.json'
            config.write_text(json.dumps({'dspark_block_size':5,'text_config':{'num_nextn_predict_layers':3,'dspark_block_size':5}}))
            report=export.materialize_pack(plan,[source],root/'pack',{'config.json':str(config)})
            exported_config=json.loads((root/'pack'/'config.json').read_text())
            self.assertEqual(exported_config['pack_revision'],{'id':report['pack_revision'],'unstamped':[]})
            self.assertNotIn('dspark_block_size',exported_config)
            self.assertNotIn('dspark_block_size',exported_config['text_config'])
            self.assertEqual(exported_config['text_config']['num_nextn_predict_layers'],0)
            self.assertEqual(report['pack_revision'],export.materialize_pack(plan,[source],root/'pack',{'config.json':str(config)})['pack_revision'])
            mapping=report['weight_map']
            self.assertTrue(all(Path(v).name==v and v.startswith('model-global-') for v in mapping.values()))
            self.assertEqual(export.validate_discovery(root/'pack',mapping),mapping)
            index=json.loads((root/'pack'/'model.safetensors.index.json').read_text())
            self.assertEqual(index['weight_map'],mapping)
            self.assertFalse(report['ready'])
            shard=root/'pack'/mapping['norm.weight']
            self.assertEqual(contract.read_header(shard)['__metadata__']['pack_revision'],report['pack_revision'])
            export.write_safetensors(root/'pack'/'model-unindexed.safetensors',{'unexpected':np.ones(1,np.float32)}, {})
            with self.assertRaisesRegex(ValueError,'discovered tensors differ'):
                export.validate_discovery(root/'pack',mapping)

    def test_lazy_imatrix_pin_provenance_and_no_stale_mix(self):
        from types import SimpleNamespace
        from orca_reader import Tensor
        class Reader:
            tensors={'blk.20.ffn_gate_exps.weight':Tensor('source',(128,128,2),'F32',0,0)}
        class Importance:
            provenance={'range_sha256':'sample123'}
            def weights(self,norm):
                self.assert_norm=norm
                return np.ones(128)
        class Calibration:
            revision='first'
            def identity(self):
                return {'revision':self.revision,'header_sha256':'header123'}
            def read_native(self,name):
                return Importance()
        calibration=Calibration()
        with tempfile.TemporaryDirectory() as directory:
            a=SimpleNamespace(catalog=None,layer=20,expert=1,projection='down',rates=None,n=28,
                              backend='cpu',seed=0,batch=1,calibration=None,imatrix=True,output=directory)
            with patch('orca_reader.OrcaReader',return_value=Reader()),patch('imatrix_reader.ImatrixReader',return_value=calibration),patch('orca_export.projection') as encode:
                result=export.export(a)
                self.assertEqual(encode.call_args.args[-1].shape,(128,))
                self.assertEqual(result.data['pin']['params']['importance_normalization'],'layer')
                native='layers.20.ffn.experts.1.w2.weight'
                self.assertEqual(result.data['importance_provenance'][native]['range_sha256'],'sample123')
                calibration.revision='second'
                with self.assertRaisesRegex(ValueError,'pin mismatch'):
                    export.export(a)

    def test_rates(self):
        rates = export.rate_plan(2, 28, {"1": {"gate": 24, "up": 32, "down": 40}})
        self.assertEqual(list(rates["1"].values()), [24, 32, 40])
        with self.assertRaises(ValueError):
            export.rate_plan(2, 28, {"3": {"down": 40}})
        with self.assertRaises(ValueError):
            export.rate_plan(2, 28, {"1": {"down": True}})

    def test_encode_layout_and_stored_scale_rounding(self):
        def fake_encoder(x, n, **kwargs):
            return np.zeros((len(x), n), np.uint16), np.zeros(len(x))
        w = np.random.default_rng(1).normal(0, .02, (128, 128)).astype(np.float32)
        with patch("pair1_cuda.encode_tiles", side_effect=fake_encoder):
            arrays, metrics = export.encode_matrix(w, 28)
        self.assertEqual(arrays["trellis"].shape, (8, 8, 28))
        self.assertEqual(arrays["suh"].dtype, np.float16)
        expected = contract.reconstruct(arrays["trellis"], arrays["suh"], arrays["svh"])
        self.assertAlmostEqual(metrics["sse"], float(np.square(expected.astype(np.float64) - w).sum()))
        self.assertEqual(metrics["payload_bits"], 128 * 128 * 28 // 16)

    def test_diagonal_calibration_matches_dense_pipeline(self):
        def codec(x,n,**kwargs):
            # Deterministic input-dependent stand-in keeps this a fast integration test.
            codes = np.zeros((len(x),n),np.uint16)
            codes[:,0] = (np.abs(x.numpy()).sum(1)*100).astype(np.uint16)
            return codes,np.zeros(len(x))
        w=np.random.default_rng(9).normal(0,.01,(128,128)).astype(np.float32)
        diagonal=np.linspace(.1,2,128)
        with patch('pair1_cuda.encode_tiles',side_effect=codec):
            vector,vm=export.encode_matrix(w,28,hessian=diagonal)
            dense,dm=export.encode_matrix(w,28,hessian=np.diag(diagonal))
        for key in vector:
            np.testing.assert_array_equal(vector[key],dense[key])
        self.assertAlmostEqual(vm['weighted_error'],dm['weighted_error'],places=10)

    def test_mixed_layer_assembly_reorders_router(self):
        with tempfile.TemporaryDirectory() as directory:
            m = export.Manifest(directory, {"test": True})
            for e, rate in enumerate((32, 24)):
                for p in export.PROJECTIONS:
                    export.write_safetensors(m.root / f"expert.{e}.{p}.safetensors", {
                        "trellis": np.full((8, 8, rate), e, np.uint16),
                        "suh": np.ones(128, np.float16), "svh": np.ones(128, np.float16)}, {})
            router = m.root / "router.safetensors"
            export.write_safetensors(router, {"layers.0.ffn.gate.weight": np.array([[1, 2], [3, 4]], np.float32),
                "layers.0.ffn.gate.bias": np.array([5, 6], np.float32),
                "layers.0.ffn.gate.bias_vl": np.array([7, 8], np.float32)}, {})
            # Small fixture uses top_k=2; real layer has top_k=6.
            original = contract.validate_routed
            with patch("pair1_contract.validate_routed", side_effect=lambda h, q, i, o, e: original(h, q, i, o, e, top_k=2)):
                export.assemble_layer(m, 0, [((24, 24, 24), [1]), ((32, 32, 32), [0])], [router], dict(contract.PAIR1_CONFIG, k=2))
            tensors = contract.read_tensors(m.root / "model-layer-00.safetensors")
            np.testing.assert_array_equal(tensors["layers.0.ffn.gate.weight"], [[3, 4], [1, 2]])
            np.testing.assert_array_equal(tensors["layers.0.ffn.gate.bias"], [6, 5])
            np.testing.assert_array_equal(tensors["layers.0.ffn.gate.bias_vl"], [8, 7])
            self.assertEqual(m.data["expert_permutation"], [1, 0])


if __name__ == "__main__":
    unittest.main()
