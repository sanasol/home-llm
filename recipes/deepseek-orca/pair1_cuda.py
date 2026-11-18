"""Batched PAIR1 least-squares encoder; CUDA is compiled lazily, CPU needs only torch.

Input is [tiles,256] in trellis order (stq.tiles_of), not row-major tiles.
Warm-up pinning gives a valid tail-biting path, not a global-optimum guarantee.
CUDA pilot (CUDA torch, nvcc and a C++ compiler required)::

    python ds-sushi-abl/test_pair1_cuda.py --cuda-pilot --tiles 64 --batch 16
"""
from pathlib import Path

import numpy as np
import torch

from pair1_reference import pair1_table, pair_fresh_bits, pack_pair_states

_EXTENSION = None


def _extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        _EXTENSION = load(
            name="sushi_pair1_cuda_v2",
            sources=[str(Path(__file__).with_suffix(".cu"))],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXTENSION


def workspace_bytes_per_tile(n, backend="torch"):
    """Conservative live search scratch, excluding input/output and compiler memory."""
    ks = pair_fresh_bits(n)
    edges = 1 << (16 - int(ks.min()))
    # Trace + two costs; torch also materializes candidates/gather/reduction temporaries.
    return (128 * edges + 16 * edges + 4096
            + (32 * 65536 if backend == "torch" else 0))


def _torch_states(x, table, ks):
    b = len(x)
    dev = x.device
    max_edges = 1 << (16 - int(ks.min()))
    back = torch.empty((128, b, max_edges), dtype=torch.uint8, device=dev)
    rows = torch.arange(b, device=dev)
    cw = torch.arange(65536, device=dev)
    pin = None
    for roll in (64, 0):
        prev = None
        for step in range(128):
            ri = (step + roll) % 128
            k, kn = int(ks[ri]), int(ks[(ri + 1) % 128])
            edges, branches = 1 << (16 - kn), 1 << kn
            loss = (table[None, :, 0] - x[:, 2 * ri, None]).square()
            loss += (table[None, :, 1] - x[:, 2 * ri + 1, None]).square()
            pred = cw >> k
            if prev is not None:
                loss += prev[:, pred]
            elif pin is not None:
                loss.masked_fill_(pred[None, :] != pin[:, None], float("inf"))
            prev, arg = loss.view(b, branches, edges).min(dim=1)
            back[ri, :, :edges] = arg
        edge = prev.argmin(1) if pin is None else pin.clone()
        states = torch.empty((b, 128), dtype=torch.int64, device=dev)
        for step in range(127, -1, -1):
            ri = (step + roll) % 128
            kr = 16 - int(ks[(ri + 1) % 128])
            state = (back[ri, rows, edge].long() << kr) | edge
            states[:, ri] = state
            edge = state >> int(ks[ri])
            if pin is None and ri == 0:
                break
        pin = edge
    return states


@torch.no_grad()
def encode_tiles(x, n, batch=64, max_workspace_mb=512, backend="auto"):
    """Return (packed uint16[T,n], SSE float64[T]) using float32 search costs.

    backend='auto' selects the fused extension for CUDA, torch otherwise. Build
    failures are explicit, never silently run a slow fallback on a GPU pilot.
    The workspace cap bounds search scratch by chunking; caller input/output and
    CUDA runtime/compiler allocations are outside that cap.
    """
    if n not in (24, 28, 32, 40):
        raise ValueError("PAIR1 CUDA encoder supports n24,n28,n32,n40")
    if backend not in ("auto", "torch", "cuda", "cuda-persistent"):
        raise ValueError("backend must be auto, torch, cuda or cuda-persistent")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if not np.isfinite(max_workspace_mb) or max_workspace_mb <= 0:
        raise ValueError("max_workspace_mb must be finite and positive")
    x = torch.as_tensor(x)
    if x.ndim != 2 or x.shape[1] != 256 or x.is_complex():
        raise ValueError("x must be a real [tiles,256] array in trellis order")
    backend = ("cuda" if x.is_cuda else "torch") if backend == "auto" else backend
    if backend.startswith("cuda") and not x.is_cuda:
        raise ValueError("CUDA backends require a CUDA input tensor")
    per_tile = workspace_bytes_per_tile(n, backend)
    limit = int(max_workspace_mb * 1024 * 1024)
    fixed_bytes = 2 << 20  # Table and device indices, independent of tile batch.
    batch = min(batch, max(0, (limit - fixed_bytes) // per_tile))
    if batch < 1:
        raise ValueError(f"workspace needs at least {fixed_bytes + per_tile} bytes for one tile")
    tab_np = pair1_table(n)
    table = torch.tensor(tab_np, device=x.device)
    ks = pair_fresh_bits(n)
    packed = np.empty((len(x), n), dtype=np.uint16)
    sse = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch):
        xb = x[start:start + batch].to(dtype=torch.float32).contiguous()
        # Squaring extreme f32 inputs can overflow even though the inputs are finite.
        if not torch.isfinite(xb).all().item() or xb.abs().max().item() > 1e15:
            raise ValueError("target magnitude exceeds safe float32 search range")
        states = (_extension().encode(xb, table, ks.tolist(), backend == "cuda-persistent")
                  if backend.startswith("cuda") else _torch_states(xb, table, ks))
        states = states.cpu().numpy().astype(np.int64, copy=False)
        packed[start:start + len(xb)] = pack_pair_states(states, n)
        delta = tab_np[states].astype(np.float64) - xb.cpu().numpy().reshape(-1, 128, 2)
        sse[start:start + len(xb)] = np.square(delta).sum(axis=(1, 2))
    return packed, sse
