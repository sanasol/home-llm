"""Sushi STQ SPEC1 trunk-bank codec (decode + trellis encode), ported from beamivalice/sushi @ d5e876e.

A trunk linear `<base>` is stored as `<base>.trellis` U16 [in/16, out/16, n], `<base>.suh` F16 [in], `<base>.svh` F16 [out]:
    W[in, out] = diag(suh) . H_in . inner . H_out . diag(svh)        (H = block-diagonal 128x128 Hadamard / sqrt(128))
`inner` is 16x16 tiles; each tile is a tail-biting trellis of 256 codewords, K = n/16 fresh bits per weight, the
codeword being the last 16 stream bits masked to the window (15). Weight = SPEC1 table[codeword].
ponytail: integer rates only (n % 16 == 0); fractional rates need per-step k, add when a pack needs K like 2.5.
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HAD = 128
W_BITS = 15


def spec1_row(n):
    with open(os.path.join(HERE, "spec1_w15.txt")) as source:
        for line in source:
            a, mult, mask = line.strip().split(",")
            if int(a) == n:
                return int(mult, 16), int(mask, 16)
    raise KeyError(n)


def spec1_table(n, w=W_BITS):
    """Decoded value of every w-bit codeword at rate n (f32, exactly as the engine's f16 arithmetic gives it)."""
    mult, mask = spec1_row(n)
    p = ((np.arange(1 << w, dtype=np.uint64) * mult) & 0xFFFFFFFF & mask).astype(np.uint32)
    lo = (p & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (p >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    s = (lo + hi).astype(np.float16).astype(np.float32) * np.float32(4096)
    return s.astype(np.float16).astype(np.float32)


def core_perm():
    """Trellis step -> row-major position inside the 16x16 tile (tensorCorePerm)."""
    out = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        for j, (r, c) in enumerate([(r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                                    (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)]):
            out[t * 8 + j] = r * 16 + c
    assert sorted(out) == list(range(256))
    return out


PERM = core_perm()


def hadamard(device="cpu", dtype=torch.float32):
    i = torch.arange(HAD)
    x = i[:, None] & i[None, :]
    bits = torch.zeros_like(x)
    for b in range(7):
        bits += (x >> b) & 1
    return ((1 - 2 * (bits % 2)).to(dtype) * 0.08838834764831845).to(device)


def rotate(m, h):
    """H_in . m . H_out for m [in, out]."""
    i, o = m.shape
    m = (h @ m.view(i // HAD, HAD, o)).view(i, o)
    return (m.view(i, o // HAD, HAD) @ h).view(i, o)


def unpack_codes(trellis):
    """U16 [..., n] -> codewords int64 [..., 256] (16-bit windows, unmasked)."""
    n = trellis.shape[-1]
    assert n % 16 == 0, "integer rates only"
    k = n // 16
    t = trellis.astype(np.uint32)
    words = t[..., 0::2] | (t[..., 1::2] << 16)                       # u32, stream is MSB-first inside each
    bits = ((words[..., None] >> np.arange(31, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8)
    bits = bits.reshape(*trellis.shape[:-1], 256, k)                   # fresh bits of each weight
    fresh = np.zeros(bits.shape[:-1], dtype=np.int64)
    for b in range(k):
        fresh = (fresh << 1) | bits[..., b]
    cw = np.zeros_like(fresh)
    steps = -(-16 // k)
    for j in range(steps):                                             # window = last 16 bits, wrapping
        cw |= np.roll(fresh, j, axis=-1) << (k * j)
    return cw & 0xFFFF


def pack_fresh(fresh, k):
    """fresh bits per weight int64 [..., 256] -> U16 [..., 16k]."""
    bits = ((fresh[..., None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint32)
    bits = bits.reshape(*fresh.shape[:-1], 8 * k, 32)
    words = (bits << np.arange(31, -1, -1, dtype=np.uint32)).sum(-1, dtype=np.uint32)
    out = np.empty((*fresh.shape[:-1], 16 * k), dtype=np.uint16)
    out[..., 0::2] = words & 0xFFFF
    out[..., 1::2] = words >> 16
    return out


def inner_from_trellis(trellis):
    """U16 [in/16, out/16, n] -> inner f32 [in, out]."""
    it, ot, n = trellis.shape
    vals = spec1_table(n)[unpack_codes(trellis) & ((1 << W_BITS) - 1)]
    tiles = np.empty_like(vals)
    tiles[..., PERM] = vals
    return np.ascontiguousarray(tiles.reshape(it, ot, 16, 16).transpose(0, 2, 1, 3)).reshape(it * 16, ot * 16)


def reconstruct(trellis, suh, svh, device="cpu"):
    """Ideal decoded weight as f32 [in, out]; excludes runtime activation and output rounding."""
    inner = torch.from_numpy(inner_from_trellis(trellis)).to(device)
    su = torch.from_numpy(suh.astype(np.float32)).to(device)
    sv = torch.from_numpy(svh.astype(np.float32)).to(device)
    return su[:, None] * rotate(inner, hadamard(device)) * sv[None, :]


def target_inner(w_in_out, suh, svh):
    """inner the trellis must approximate for a dense target [in, out] under fixed scales."""
    su = torch.from_numpy(suh.astype(np.float32)).to(w_in_out.device)
    sv = torch.from_numpy(svh.astype(np.float32)).to(w_in_out.device)
    return rotate(w_in_out / su[:, None] / sv[None, :], hadamard(w_in_out.device))


def tiles_of(inner):
    """inner [in, out] -> x [tiles, 256] in trellis-step order (tile index = in_tile * out_tiles + out_tile)."""
    i, o = inner.shape
    t = inner.view(i // 16, 16, o // 16, 16).permute(0, 2, 1, 3).reshape(-1, 256)
    return t[:, torch.from_numpy(PERM).to(inner.device)]


@torch.no_grad()
def encode_tiles(x, n, batch=4096):
    """Least-squares tail-biting trellis search (the engine encoder's two-pass pinning), batched.

    Float32 least-squares variant, NOT byte-equivalent to Sushi's float16-cost encoder.
    x [T, 256] f32 on the search device -> (U16 numpy [T, n], squared error per tile numpy [T]).
    A codeword is (j << kr) | edge, j being its top k bits; the previous step's edge is
    codeword >> k = (j << (kr - k)) | (edge >> k), so the predecessor cost broadcasts over the low k bits of edge.
    """
    assert n % 16 == 0, "integer rates only"
    k, dev = n // 16, x.device
    if not 1 <= k <= 7:
        raise ValueError("This search supports integer K1 through K7 only")
    kr = W_BITS - k                       # bits of a codeword that survive into the next one ("edge")
    E, J, L = 1 << kr, 1 << k, 1 << (kr - k)
    tab = torch.from_numpy(spec1_table(n)).to(dev)
    t1 = tab.view(1, J, E)
    t2 = (tab * tab).view(1, J, E)
    pred = (((torch.arange(J, device=dev)[:, None] << kr) | torch.arange(E, device=dev)[None, :]) >> k).view(1, J, E)
    out = np.empty((x.shape[0], n), dtype=np.uint16)
    sse = np.empty(x.shape[0], dtype=np.float64)
    for s in range(0, x.shape[0], batch):
        xb = x[s:s + batch]
        B = xb.shape[0]
        rows = torch.arange(B, device=dev)
        back = torch.empty((256, B, E), dtype=torch.uint8, device=dev)
        m2x = (-2 * xb).T.contiguous()                                    # [256, B]
        pin = None
        for roll in (128, 0):             # pass 0: free start half a tile away, to learn the wrap state; pass 1: pinned
            prev = None
            for step in range(256):
                ri = (step + roll) & 255
                d = torch.addcmul(t2, t1, m2x[ri].view(B, 1, 1))          # (t - x)^2 - x^2, [B, J, E]
                if prev is not None:
                    d.view(B, J, L, J).add_(prev.view(B, J, L, 1))
                elif pin is not None:
                    d.masked_fill_(pred != pin.view(B, 1, 1), float("inf"))
                prev, arg = d.min(dim=1)
                back[ri] = arg
            edge = prev.argmin(dim=1) if pin is None else pin
            states = torch.empty((B, 256), dtype=torch.int64, device=dev)
            for step in range(255, -1, -1):
                ri = (step + roll) & 255
                state = (back[ri][rows, edge].to(torch.int64) << kr) | edge
                states[:, ri] = state
                edge = state >> k
                if pin is None and ri == 0:
                    break
            pin = edge
        sse[s:s + B] = ((tab[states] - xb) ** 2).sum(1, dtype=torch.float64).cpu().numpy()
        out[s:s + B] = pack_fresh((states & (J - 1)).cpu().numpy(), k)
    return out, sse


E4M3 = np.zeros(256, dtype=np.float32)
for _c in range(256):
    _e, _m = (_c >> 3) & 15, _c & 7
    _v = (_m / 8) * 2.0 ** -6 if _e == 0 else (1 + _m / 8) * 2.0 ** (_e - 7)
    E4M3[_c] = float("nan") if (_c & 0x7F) == 0x7F else (-_v if _c & 0x80 else _v)


def fp8_dense(codes, scale):
    """Release FP8: e4m3 codes U8 [out, in] with one e8m0 scale per 32x32 block -> f32 [out, in]."""
    o, i = codes.shape
    if np.any(scale == 255) or np.any((codes & 0x7F) == 0x7F):
        raise ValueError("Non-finite FP8 weight or E8M0 scale sentinel")
    s = np.exp2(scale.astype(np.float32) - 127)
    s = np.repeat(np.repeat(s, 32, axis=0)[:o], 32, axis=1)[:, :i]
    return E4M3[codes] * s
