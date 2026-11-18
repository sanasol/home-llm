"""CPU checks by default; --cuda-pilot explicitly requires a working CUDA toolchain."""
import argparse
import json
import os
from pathlib import Path
import time
import unittest

import numpy as np
import torch

import pair1_cuda as gpu
import pair1_reference as ref
import pair1_contract as contract
from stq import PERM


class Pair1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rates_reference_and_decode(self):
        for n in (24, 28, 32, 40):
            with self.subTest(n=n):
                x = np.random.default_rng(n).normal(size=(2, 256)).astype(np.float32)
                packed, cost = gpu.encode_tiles(x, n, batch=2)
                expected, expected_cost = ref.encode_tiles(x, n)
                decoded = contract.decode_tiles(packed).astype(np.float32)[:, PERM]
                np.testing.assert_array_equal(decoded, ref.decode_tiles(packed)[:, PERM])
                measured = np.square(decoded.astype(np.float64) - x).sum(axis=1)
                np.testing.assert_allclose(cost, measured, atol=1e-10)
                # FP32 DP need not choose identical bytes to the FP64 oracle.
                np.testing.assert_allclose(cost, expected_cost, rtol=2e-5, atol=2e-4)
                again, again_cost = gpu.encode_tiles(x, n, batch=1)
                np.testing.assert_array_equal(again, packed)
                np.testing.assert_array_equal(again_cost, cost)
                self.assertEqual(packed.dtype, np.uint16)
                self.assertEqual(packed.shape, (2, n))

    def test_exactly_representable(self):
        for n in (24, 28, 32, 40):
            packed = np.random.default_rng(n).integers(0, 65536, (1, n), dtype=np.uint16)
            x = ref.decode_tiles(packed)[:, PERM]
            encoded, cost = gpu.encode_tiles(x, n)
            np.testing.assert_array_equal(ref.decode_tiles(encoded)[:, PERM], x)
            np.testing.assert_array_equal(cost, [0.0])

    def test_public_sushi_fixture(self):
        path = Path(os.environ.get("PAIR1_FIXTURE", "/tmp/sushi-src/src/exl3/fixtures/exl3_pair1_tiles.safetensors"))
        if not path.is_file():
            self.skipTest("set PAIR1_FIXTURE to public exl3_pair1_tiles.safetensors")
        tensors = contract.read_tensors(path)
        for n in (24, 28, 32, 40):
            expected = tensors[f"decoded_n{n}"].view(np.float16).astype(np.float32)
            np.testing.assert_array_equal(contract.decode_tiles(tensors[f"trellis_n{n}"]), expected)
            packed, cost = gpu.encode_tiles(expected[:1, PERM], n)
            np.testing.assert_array_equal(contract.decode_tiles(packed), expected[:1])
            np.testing.assert_array_equal(cost, [0.0])

    def test_budget_and_validation(self):
        x = torch.zeros((1, 256))
        for kwargs in ({"n": 26}, {"n": 28, "batch": 0},
                       {"n": 28, "backend": "bad"}, {"n": 28, "backend": "cuda"},
                       {"n": 28, "max_workspace_mb": 0.001}):
            with self.assertRaises(ValueError):
                gpu.encode_tiles(x, **kwargs)
        for bad in (torch.full((1, 256), float("nan")), torch.zeros(256),
                    torch.full((1, 256), 1e30)):
            with self.assertRaises(ValueError):
                gpu.encode_tiles(bad, 28)
        packed, cost = gpu.encode_tiles(torch.empty((0, 256)), 28)
        self.assertEqual(packed.shape, (0, 28))
        self.assertEqual(cost.shape, (0,))
        # The cap forces one tile per chunk, regardless of the requested batch.
        budget = (gpu.workspace_bytes_per_tile(28) + (2 << 20)) / 1024 ** 2
        bounded, _ = gpu.encode_tiles(torch.zeros((2, 256)), 28, batch=10,
                                      max_workspace_mb=budget)
        self.assertEqual(bounded.shape, (2, 28))


def cuda_verify():
    """All-64-tile CPU parity plus ties, chunking and strided input checks."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requires an available GPU")
    torch.set_num_threads(1)
    for n in (24, 28, 32, 40):
        x = np.random.default_rng(n + 100).normal(size=(64, 256)).astype(np.float32)
        x[:2] = 0  # Repeated zero costs exercise deterministic first-on-ties.
        expected, expected_cost = gpu.encode_tiles(x, n, batch=8, backend="torch")
        strided = torch.empty((64, 512), device="cuda")[:, ::2]
        strided.copy_(torch.from_numpy(x))
        assert not strided.is_contiguous()
        for backend in ("cuda", "cuda-persistent"):
            packed, cost = gpu.encode_tiles(strided, n, batch=17, backend=backend)
            np.testing.assert_array_equal(packed, expected)
            np.testing.assert_array_equal(cost, expected_cost)
            single, single_cost = gpu.encode_tiles(strided[:5], n, batch=1, backend=backend)
            np.testing.assert_array_equal(single, expected[:5])
            np.testing.assert_array_equal(single_cost, expected_cost[:5])
            decoded = contract.decode_tiles(packed).astype(np.float32)[:, PERM]
            measured = np.square(decoded.astype(np.float64) - x).sum(1)
            np.testing.assert_array_equal(measured, cost)
            # Every GPU output is a valid circular stream under the independent decoder.
            representable, zero = gpu.encode_tiles(torch.from_numpy(decoded[:3]).cuda(), n,
                                                  backend=backend)
            np.testing.assert_array_equal(contract.decode_tiles(representable).astype(np.float32)[:, PERM],
                                          decoded[:3])
            np.testing.assert_array_equal(zero, np.zeros(3))
            print(json.dumps({"verified": True, "n": n, "backend": backend,
                              "cpu_parity_tiles": 64, "strided": True,
                              "ties": True, "single_vs_chunk": True}), flush=True)


def cuda_pilot(tiles, batch, backend="cuda", workspace_mb=512):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pilot requested but torch has no available CUDA device")
    if tiles < 1 or batch < 1:
        raise ValueError("tiles and batch must be positive")
    torch.set_num_threads(1)
    device = torch.cuda.current_device()
    for n in (24, 28, 32, 40):
        x = torch.from_numpy(np.random.default_rng(n).normal(size=(tiles, 256)).astype(np.float32)).cuda()
        # Separate compilation/warm-up from measured end-to-end encode time.
        gpu.encode_tiles(x[:1], n, batch=1, backend=backend, max_workspace_mb=workspace_mb)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        packed, cost = gpu.encode_tiles(x, n, batch=batch, backend=backend, max_workspace_mb=workspace_mb)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        cpu_packed, cpu_cost = gpu.encode_tiles(x[:min(2, tiles)].cpu(), n, batch=1)
        np.testing.assert_array_equal(packed[:len(cpu_packed)], cpu_packed)
        np.testing.assert_allclose(cost[:len(cpu_cost)], cpu_cost, rtol=1e-6, atol=1e-6)
        decoded = contract.decode_tiles(packed).astype(np.float32)[:, PERM]
        np.testing.assert_array_equal(decoded, ref.decode_tiles(packed)[:, PERM])
        measured = np.square(decoded.astype(np.float64) - x.cpu().numpy()).sum(axis=1)
        np.testing.assert_allclose(cost, measured, rtol=1e-12, atol=1e-10)
        # Exact representable input catches pin/trace errors independent of random SSE.
        recovered, recovered_cost = gpu.encode_tiles(torch.from_numpy(decoded[:2]).cuda(), n, batch=batch,
                                                      backend=backend, max_workspace_mb=workspace_mb)
        np.testing.assert_array_equal(ref.decode_tiles(recovered)[:, PERM], decoded[:2])
        np.testing.assert_array_equal(recovered_cost, np.zeros(len(recovered_cost)))
        print(json.dumps({"device": torch.cuda.get_device_name(device), "n": n,
                          "tiles": tiles, "batch": batch, "backend": backend,
                          "workspace_mb": workspace_mb,
                          "effective_batch": min(batch, int((workspace_mb * 1024**2 - (2 << 20))
                                                   // gpu.workspace_bytes_per_tile(n, backend))),
                          "seconds": elapsed,
                          "weights_per_second": tiles * 256 / elapsed,
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "mean_sse": float(cost.mean()), "torch": torch.__version__,
                          "cuda": torch.version.cuda}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-pilot", action="store_true")
    parser.add_argument("--cuda-verify", action="store_true")
    parser.add_argument("--tiles", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--backend", choices=("cuda", "cuda-persistent"), default="cuda")
    parser.add_argument("--workspace-mb", type=float, default=512)
    args, extra = parser.parse_known_args()
    if args.cuda_verify:
        cuda_verify()
    elif args.cuda_pilot:
        cuda_pilot(args.tiles, args.batch, args.backend, args.workspace_mb)
    else:
        unittest.main(argv=[__file__, *extra])
