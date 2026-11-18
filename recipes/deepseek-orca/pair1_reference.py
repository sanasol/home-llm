"""CPU PAIR1 reference for Sushi d5e876e, pair1-w16-v1 (digest c089b4eb).

A 256-value tile has 128 JOINT pair observations. At pair j its window ends
at E_j=floor((2j+2)*n/16), with q_j=E_j-E_{j-1} fresh bits. Both
values come from that same odd-weight window, never from separate scalar
windows. A path obeys c_j=((c_{j-1} << q_j)|f_j)&65535 and minimizes
sum_j ||x_j - table[c_j,:]||^2.

For an entering overlap r, initialize D_0(c)=loss_0(c) if c>>q_0=r,
otherwise infinity. Then D_j(c)=loss_j(c)+min_h D_{j-1}((h<<(w-q_j))
| (c>>q_j)). Finish only at c whose low w-q_0 bits equal r. The min is
first-on-ties. The default encoder learns r from a free half-tile-rolled
pass, then solves the pinned problem exactly in float64. This makes the
packed tail bite exactly, but does NOT prove a globally optimal circular
path. exact_tail_biting is an exhaustive narrow synthetic-table oracle.

Stream bits are MSB first within uint32 words; each word is stored low
uint16 then high uint16. No scales, Hadamard transform, LDLQ, GGUF reader,
or production encoder integration is implemented here. Unlike Sushi's
scalar encoder, PAIR1 scores two values per branch and uses w16 only.
"""
from functools import lru_cache

import numpy as np

from stq import PERM


TABLE_NAME = "pair1-w16-v1"
TABLE_DIGEST = "c089b4eb"
SOURCE_SHA = "d5e876edcb1538a347041b1b7cc28a0e6e8f0af6"
# Frozen PAIR1_ROWS: n=24,26,...,64; n38's swapped mask0 is intentional.
PAIR1_ROWS = (
    (0x58F41BF3, 3, 0x8ED78C13, 0x8EEE8C03),
    (0x58F41BF3, 3, 0x8ED78C13, 0x8EEE8C03),
    (0x7AA2BE15, 3, 0x8ED78C13, 0x8ECE8C03),
    (0x7B229E11, 3, 0x8ED78C13, 0x8ECE8C03),
    (0x3B22BE15, 3, 0x8ED78C13, 0x8ECE8C03),
    (0x46C63E17, 19, 0x8F9A8E6E, 0x8E258FFE),
    (0x46C63E17, 19, 0x8F9A8E6E, 0x8E258FFE),
    (0x46C63E17, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0x46C63E17, 19, 0x8F9A8E6E, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0xF75D60FF, 19, 0x8E6E8F9A, 0x8E258FFE),
    (0x82439F67, 19, 0x8E6E8F9A, 0x8E258FAE),
    (0xE2C7FC0D, 19, 0x8F408FB0, 0x8F918F89),
    (0xE2C7FC0D, 19, 0x8F408FA0, 0x8F118FC9),
    (0xE2C7FC0D, 19, 0x8F408FA0, 0x8F118FC9),
    (0xE2C7FC0D, 19, 0x8F408FA0, 0x8F118FC9),
)


def _rate(n):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n % 2 or not 24 <= n <= 64:
        raise ValueError("PAIR1 needs even halfword count n=24..64 (K=n/16)")
    return int(n)


def pair_fresh_bits(n):
    """Fresh bits of each JOINT pair; n28 alternates 3,4, not scalar K2."""
    n = _rate(n)
    return np.diff(np.arange(129, dtype=np.int64) * n // 8)


@lru_cache(maxsize=21, typed=True)
def pair1_table(n):
    """All 65536 decoded pairs, float32 exact to the engine's half arithmetic."""
    n = _rate(n)
    mult, rot, mask0, mask1 = PAIR1_ROWS[(n - 24) // 2]
    p = ((np.arange(65536, dtype=np.uint64) * mult) & 0xFFFFFFFF).astype(np.uint32)
    rotated = (p << np.uint32(rot)) | (p >> np.uint32(32 - rot))
    values = []
    for halves in (p & mask0, rotated & mask1):
        lo = (halves & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        hi = (halves >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
        summed = (lo + hi).astype(np.float16).astype(np.float32)
        values.append((summed * np.float32(4096)).astype(np.float16).astype(np.float32))
    table = np.stack(values, axis=-1)
    table.flags.writeable = False
    return table


def unpack_codes(packed):
    """U16 [...,n] -> all 256 circular 16-bit scalar endpoint windows."""
    packed = np.asarray(packed)
    if packed.ndim < 1 or packed.dtype != np.uint16:
        raise ValueError("packed tiles must be uint16 [...,n]")
    n = _rate(packed.shape[-1])
    t = packed.astype(np.uint32)
    words = t[..., 0::2] | (t[..., 1::2] << 16)
    bits = ((words[..., None] >> np.arange(31, -1, -1, dtype=np.uint32)) & 1).astype(np.uint16)
    bits = bits.reshape(*packed.shape[:-1], 16 * n)
    ends = np.arange(1, 257) * n // 16
    at = (ends[:, None] - 16 + np.arange(16)) % (16 * n)
    return np.sum(bits[..., at].astype(np.int64) << np.arange(15, -1, -1), axis=-1)


def pack_fresh(fresh, n):
    """Pair fresh values [...,128] -> U16 [...,n], n is NOT integer K."""
    q = pair_fresh_bits(n)
    fresh = np.asarray(fresh)
    if fresh.ndim < 1 or fresh.shape[-1] != 128 or fresh.dtype.kind not in "iu":
        raise ValueError("fresh values must be integer [...,128]")
    if np.any(fresh < 0) or np.any(fresh >= (1 << q)):
        raise ValueError("fresh value exceeds its pair's bit budget")
    bits = np.concatenate([((fresh[..., j, None] >> np.arange(k - 1, -1, -1)) & 1)
                           for j, k in enumerate(q)], axis=-1).astype(np.uint32)
    words = np.sum(bits.reshape(*fresh.shape[:-1], n // 2, 32)
                   << np.arange(31, -1, -1, dtype=np.uint32), axis=-1, dtype=np.uint32)
    out = np.empty((*fresh.shape[:-1], n), dtype=np.uint16)
    out[..., 0::2] = words & 0xFFFF
    out[..., 1::2] = words >> 16
    return out


def pack_pair_states(states, n):
    states = np.asarray(states)
    if states.ndim < 1 or states.shape[-1] != 128 or states.dtype.kind not in "iu":
        raise ValueError("states must be integer [...,128]")
    if np.any(states < 0) or np.any(states > 65535):
        raise ValueError("PAIR1 codewords must fit in 16 bits")
    q = pair_fresh_bits(n)
    mask = (1 << (16 - q)) - 1
    if np.any((states >> q) != (np.roll(states, 1, axis=-1) & mask)):
        raise ValueError("states do not form a tail-biting PAIR1 path")
    return pack_fresh(states & ((1 << q) - 1), n)


def decode_tiles(packed):
    """U16 [...,n] -> row-major float32 [...,256], no Hadamard or scales."""
    packed = np.asarray(packed)
    codes = unpack_codes(packed)
    values = pair1_table(packed.shape[-1])[codes[..., 1::2]].reshape(*codes.shape[:-1], 256)
    out = np.empty_like(values)
    out[..., PERM] = values
    return out


def viterbi_pairs(xpairs, ks, table, pin=None):
    """Joint-pair float64 DP, free or pinned; returns (codeword path, SSE).

    Synthetic power-of-two tables are accepted for exhaustive small tests.
    Production PAIR1 callers must use pair1_table(n), with exactly 65536 rows.
    """
    x = np.asarray(xpairs, dtype=np.float64)
    table = np.asarray(table, dtype=np.float64)
    ks = np.asarray(ks)
    states = len(table)
    w = states.bit_length() - 1
    if (table.shape != (states, 2) or states < 2 or states > 65536 or states != 1 << w
            or x.ndim != 2 or x.shape[1] != 2 or not len(x)
            or ks.shape != (len(x),) or ks.dtype.kind not in "iu"
            or np.any(ks < 1) or np.any(ks >= w)):
        raise ValueError("invalid pair trellis shapes, table size, or fresh-bit widths")
    if not np.isfinite(x).all() or not np.isfinite(table).all():
        raise ValueError("trellis targets and table must be finite")
    k0 = int(ks[0])
    if pin is not None and (not isinstance(pin, (int, np.integer)) or not 0 <= pin < 1 << (w - k0)):
        raise ValueError("pin exceeds entering overlap register")
    prev = np.sum((table - x[0]) ** 2, axis=1)
    if pin is not None:
        prev[np.arange(states) >> k0 != pin] = np.inf
    back = [None]
    for j in range(1, len(x)):
        k = int(ks[j])
        s = 1 << (w - k)
        grouped = prev.reshape(1 << k, s)
        arg = grouped.argmin(axis=0)
        best = grouped[arg, np.arange(s)]
        prev = np.repeat(best, 1 << k) + np.sum((table - x[j]) ** 2, axis=1)
        back.append(arg.astype(np.uint16))
    if pin is not None:
        prev[(np.arange(states) & ((1 << (w - k0)) - 1)) != pin] = np.inf
    cur = int(prev.argmin())
    cost = float(prev[cur])
    if not np.isfinite(cost):
        raise ValueError("no finite path satisfies the pin")
    path = np.empty(len(x), dtype=np.int64)
    path[-1] = cur
    for j in range(len(x) - 1, 0, -1):
        k = int(ks[j])
        low = cur >> k
        cur = (int(back[j][low]) << (w - k)) | low
        path[j - 1] = cur
    return path, cost


def exact_tail_biting(xpairs, ks, table, max_registers=256):
    """Global circular optimum by every pin; deliberately bounded test oracle."""
    w = len(table).bit_length() - 1
    registers = 1 << (w - int(ks[0]))
    if registers > max_registers:
        raise ValueError("exact circular search is exponential; raise max_registers explicitly")
    best_path, best = None, float("inf")
    for pin in range(registers):
        path, cost = viterbi_pairs(xpairs, ks, table, pin)
        if cost < best:
            best_path, best = path, cost
    return best_path, best


def encode_tiles(x, n, batch=1):
    """CPU targets [T,256] in TRELLIS order -> (U16 [T,n], f64 SSE [T]).

    Tiles are processed singly to bound memory; batch is accepted for API
    parity, not used to allocate a batched state tensor. Float64 costs are
    intentionally not byte-equivalent to the upstream scalar half-cost search.
    """
    n = _rate(n)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 256 or not np.isfinite(x).all():
        raise ValueError("targets must be finite CPU [T,256] in trellis order")
    if not isinstance(batch, (int, np.integer)) or batch < 1:
        raise ValueError("batch must be a positive integer")
    table = pair1_table(n)
    ks = pair_fresh_bits(n)
    packed = np.empty((len(x), n), dtype=np.uint16)
    costs = np.empty(len(x), dtype=np.float64)
    for i, tile in enumerate(x):
        pairs = tile.reshape(128, 2)
        warm, _ = viterbi_pairs(np.roll(pairs, -64, axis=0), np.roll(ks, -64), table)
        pin = int(warm[64]) >> int(ks[0])
        path, costs[i] = viterbi_pairs(pairs, ks, table, pin)
        packed[i] = pack_pair_states(path, n)
    return packed, costs
