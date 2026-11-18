"""CPU calibration primitives, not a complete/model-quality-validated encoder.

Matrices use public [input, output] orientation. Search callbacks own PAIR1
packing and trellis order; this module hands them row-major 16x16 tiles.
"""
import math
import numpy as np

HAD = 128
TILE = 16
SIGMA_REG = 0.025
CODEBOOK_SCALE = 1.24371088


def _matrix(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return x


def _hessian(h, n):
    h = _matrix(h, "Hessian")
    if h.shape != (n, n) or not np.allclose(h, h.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Hessian must be symmetric and match input width")
    if np.linalg.eigvalsh(h).min() < -1e-9 * max(1.0, np.abs(h).max()):
        raise ValueError("Hessian must be positive semidefinite")
    return h


def hadamard_axis(x, axis):
    """Normalized, block-diagonal H128; no random signs are implicit."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or x.shape[axis] % HAD:
        raise ValueError("Hadamard axis must be finite and a multiple of 128")
    a = np.moveaxis(x, axis, -1).copy()
    shape = a.shape
    a = a.reshape(*shape[:-1], shape[-1] // HAD, HAD)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        blocks = a.reshape(*a.shape[:-1], HAD // (2 * stride), 2, stride)
        left, right = blocks[..., 0, :].copy(), blocks[..., 1, :].copy()
        blocks[..., 0, :] = left + right
        blocks[..., 1, :] = left - right
    return np.moveaxis((a / math.sqrt(HAD)).reshape(shape), -1, axis)


def _scales(w, suh, svh):
    su, sv = np.asarray(suh, dtype=np.float64), np.asarray(svh, dtype=np.float64)
    if su.shape != (w.shape[0],) or sv.shape != (w.shape[1],):
        raise ValueError("scale shapes must match public [input, output]")
    if not np.isfinite(su).all() or not np.isfinite(sv).all():
        raise ValueError("scales must be finite")
    return su, sv


def target_inner(w, suh, svh):
    """Invert W = diag(suh) H128 inner H128 diag(svh)."""
    w = _matrix(w, "weight")
    su, sv = _scales(w, suh, svh)
    if np.any(su == 0) or np.any(sv == 0):
        raise ValueError("target inversion needs nonzero scales")
    return hadamard_axis(hadamard_axis(w / su[:, None] / sv[None, :], 0), 1)


def reconstruct(inner, suh, svh):
    inner = _matrix(inner, "inner")
    su, sv = _scales(inner, suh, svh)
    return su[:, None] * hadamard_axis(hadamard_axis(inner, 0), 1) * sv[None, :]


def regularize(w, sign_in, sign_out, out_scales=True):
    """Sushi RMS scale recipe in f64, explicit signs; not f32/byte parity.

    Returns (inner, suh, svh). No global-codebook-scale search or F16 rounding.
    For actual storage round scales to F16, then call target_inner again.
    """
    w = _matrix(w, "weight")
    su, sv = _scales(w, sign_in, sign_out)
    if not np.isin(su, [-1, 1]).all() or not np.isin(sv, [-1, 1]).all():
        raise ValueError("sign vectors must contain only -1 and +1")
    rms = np.sqrt(np.mean(w * w, axis=0))
    mean = rms.mean()
    relative = rms / mean if mean > 1e-30 else np.zeros_like(rms)
    if out_scales:
        sv = sv * np.where(np.abs(relative) < 1e-30, 0.1, relative) + 1e-10
    rotated = hadamard_axis(w / sv[None, :], 1)
    rms_in = np.sqrt(np.mean(rotated * rotated, axis=1))
    su = su * np.where(rms_in < 1e-30, 0.1, rms_in) / -CODEBOOK_SCALE + 1e-10
    # Keep scales invertible even for a zero column; its target is still exactly zero.
    return target_inner(w, su, sv), su, sv


def weighted_error(w, decoded, h):
    """Sum of squared output errors: tr((W-Q)^T H (W-Q)), H = X^T X.

    Divide by captured row count separately for mean-per-token loss. Do not
    compare normalized and unnormalized Hessians as expert allocation costs.
    """
    w, decoded = _matrix(w, "weight"), _matrix(decoded, "decoded")
    if w.shape != decoded.shape:
        raise ValueError("weight and decoded shapes differ")
    h = _hessian(h, w.shape[0])
    e = w - decoded
    return max(0.0, float(np.sum(e * (h @ e))))


def rotated_hessian(h, suh):
    """H_inner = H128 diag(suh) H_public diag(suh) H128."""
    su = np.asarray(suh, dtype=np.float64)
    if su.ndim != 1 or not np.isfinite(su).all():
        raise ValueError("suh must be a finite vector")
    h = _hessian(h, len(su))
    return hadamard_axis(hadamard_axis(h * su[:, None] * su[None, :], 0), 1)


def prepare_ldl(h):
    """Return (strict block L, block D, damped H), H=(I+L)D(I+L)^T.

    Blocks are 16 input rows, as in Sushi. Empty diagonals become one, then
    sigma_reg*mean(diag) is added, with up to ten fixed retry bumps.
    """
    h = _matrix(h, "Hessian")
    h = _hessian(h, h.shape[0]).copy()
    n = len(h)
    if n % TILE:
        raise ValueError("LDL geometry requires multiples of 16")
    diag = np.diag_indices(n)
    h[diag] = np.where(h[diag] == 0, 1.0, h[diag])
    h[diag] += SIGMA_REG * h[diag].mean()
    damped_mean = h[diag].mean()
    for attempt in range(11):
        try:
            chol = np.linalg.cholesky(h)
            break
        except np.linalg.LinAlgError:
            if attempt == 10:
                raise ValueError("Hessian is not positive definite after damping") from None
            h[diag] += 2 * SIGMA_REG * damped_mean
    l, d = np.zeros_like(h), np.zeros_like(h)
    for start in range(0, n, TILE):
        stop = start + TILE
        block = chol[start:stop, start:stop]
        d[start:stop, start:stop] = block @ block.T
        l[stop:, start:stop] = np.linalg.solve(block.T, chol[stop:, start:stop].T).T
    return l, d, h


def ldlq(inner, l, encode):
    """Reverse 16-row LDLQ with an injectable PAIR1 CPU/CUDA tile search.

    encode(row_major_tiles[output_tiles,16,16]) -> (packed[output_tiles,n],
    decoded[output_tiles,16,16]). Returns packed[input_tiles,output_tiles,n]
    and decoded [input,output]. Tile callback must return actual decoded codes,
    not the compensated target. F64 feedback is not Sushi's f32 operation order.
    """
    inner, l = _matrix(inner, "inner"), _matrix(l, "L")
    rows, cols = inner.shape
    if rows % TILE or cols % TILE or l.shape != (rows, rows):
        raise ValueError("LDLQ needs 16x16 geometry and an input-width L")
    if any(np.any(l[s:s + TILE, s:]) for s in range(0, rows, TILE)):
        raise ValueError("L must be strictly block lower triangular")
    decoded = np.zeros_like(inner)
    packed = None
    for start in range(rows - TILE, -1, -TILE):
        stop = start + TILE
        compensated = inner[start:stop] + l[stop:, start:stop].T @ (inner[stop:] - decoded[stop:])
        tiles = compensated.reshape(TILE, cols // TILE, TILE).transpose(1, 0, 2).copy()
        codes, rec = encode(tiles)
        codes, rec = np.asarray(codes), np.asarray(rec, dtype=np.float64)
        if (codes.ndim != 2 or codes.shape[0] != cols // TILE or codes.dtype != np.uint16
                or not 16 <= codes.shape[1] <= 128 or codes.shape[1] % 2
                or rec.shape != tiles.shape or not np.isfinite(rec).all()):
            raise ValueError("encoder returned invalid packed/decoded tiles")
        if packed is None:
            packed = np.empty((rows // TILE, *codes.shape), dtype=np.uint16)
        if codes.shape != packed.shape[1:]:
            raise ValueError("encoder changed rate within a tensor")
        packed[start // TILE] = codes
        decoded[start:stop] = rec.transpose(1, 0, 2).reshape(TILE, cols)
    return packed, decoded


def _diagonal(diagonal, n):
    diagonal = np.asarray(diagonal, dtype=np.float64)
    if diagonal.shape != (n,) or not np.isfinite(diagonal).all() or np.any(diagonal < 0):
        raise ValueError("native diagonal must be finite, nonnegative and match input width")
    return diagonal


def weighted_error_diagonal(w, decoded, diagonal):
    """Exact native-diagonal proxy loss, no dense matrix or output-scale omission."""
    w, decoded = _matrix(w, "weight"), _matrix(decoded, "decoded")
    if decoded.shape != w.shape:
        raise ValueError("weight and decoded shapes differ")
    diagonal = _diagonal(diagonal, w.shape[0])
    return float(diagonal @ np.sum((w - decoded) ** 2, axis=1))


def prepare_diagonal_ldl(diagonal, suh):
    """Strict 16-row LDL factors [input/128,128,128] for native diagonal H.

    H128 diag(suh**2 * diagonal) H128 is exactly block diagonal128. Preserve
    dense prepare_ldl's GLOBAL damping mean (not a separate mean per block).
    Positive-semidefiniteness follows analytically; no cubic eigensystem check.
    Storage O(input*128), factor work O(input*128**2), rather than dense cubic.
    """
    su = np.asarray(suh, dtype=np.float64)
    if su.ndim != 1 or not len(su) or len(su) % HAD or not np.isfinite(su).all():
        raise ValueError("suh must be finite and have positive H128-aligned length")
    diagonal = _diagonal(diagonal, len(su))
    with np.errstate(over="ignore", invalid="ignore"):
        native = (diagonal * su ** 2).reshape(-1, HAD)
    if not np.isfinite(native).all():
        raise ValueError("scaled Hessian overflows")
    h = native[:, :, None] * np.eye(HAD)[None, :, :]
    h = hadamard_axis(hadamard_axis(h, 1), 2)
    ix = np.arange(HAD)
    h[:, ix, ix] = np.where(h[:, ix, ix] == 0, 1.0, h[:, ix, ix])
    h[:, ix, ix] += SIGMA_REG * h[:, ix, ix].mean()
    damped_mean = h[:, ix, ix].mean()
    for attempt in range(11):
        try:
            chol = np.linalg.cholesky(h)
            break
        except np.linalg.LinAlgError:
            if attempt == 10:
                raise ValueError("diagonal Hessian failed damped Cholesky") from None
            h[:, ix, ix] += 2 * SIGMA_REG * damped_mean
    l = np.zeros_like(h)
    for start in range(0, HAD, TILE):
        stop = start + TILE
        l[:, stop:, start:stop] = np.linalg.solve(
            chol[:, start:stop, start:stop].transpose(0, 2, 1),
            chol[:, stop:, start:stop].transpose(0, 2, 1)).transpose(0, 2, 1)
    return l


def ldlq_blocks(inner, l_blocks, encode):
    """Same LDLQ callback/order as dense ldlq, no cross-H128 feedback exists."""
    inner = _matrix(inner, "inner")
    blocks = np.asarray(l_blocks, dtype=np.float64)
    if inner.shape[0] % HAD or blocks.shape != (inner.shape[0] // HAD, HAD, HAD):
        raise ValueError("LDL blocks must match H128 input geometry")
    decoded, packed_blocks = np.empty_like(inner), []
    expected_shape = None
    for group in range(len(blocks) - 1, -1, -1):
        start = group * HAD
        packed, rec = ldlq(inner[start:start + HAD], blocks[group], encode)
        if expected_shape is not None and packed.shape != expected_shape:
            raise ValueError("encoder changed rate between H128 blocks")
        expected_shape = packed.shape
        packed_blocks.append(packed)
        decoded[start:start + HAD] = rec
    return np.concatenate(packed_blocks[::-1], axis=0), decoded


def ldlq_blocks_parallel(inner, l_blocks, encode):
    """Eight reverse16-row waves, independent H128 blocks in one tile batch.

    Same f64 dot-product feedback as ldlq_blocks; batch composition changes, so
    a codec must be tile-independent. Output is ordinary [IT,OT,n] pack order.
    """
    inner = _matrix(inner, "inner")
    rows, cols = inner.shape
    blocks = np.asarray(l_blocks, dtype=np.float64)
    if (rows % HAD or cols % TILE or blocks.shape != (rows // HAD, HAD, HAD)
            or not np.isfinite(blocks).all()):
        raise ValueError("parallel LDLQ needs finite matching H128 blocks and 16 output grid")
    if any(np.any(blocks[:, s:s + TILE, s:]) for s in range(0, HAD, TILE)):
        raise ValueError("L must be strictly block lower triangular")
    groups, ot = rows // HAD, cols // TILE
    target = inner.reshape(groups, HAD, cols)
    decoded = np.zeros_like(target)
    packed = None
    for start in range(HAD - TILE, -1, -TILE):
        stop = start + TILE
        # Keep each BLAS call identical to the sequential reference, avoiding
        # changed accumulation from a batched matmul implementation.
        compensated = np.stack([target[g, start:stop] + blocks[g, stop:, start:stop].T
                                @ (target[g, stop:] - decoded[g, stop:])
                                for g in range(groups)])
        tiles = compensated.reshape(groups, TILE, ot, TILE).transpose(0, 2, 1, 3).reshape(-1, TILE, TILE)
        codes, rec = encode(tiles)
        codes, rec = np.asarray(codes), np.asarray(rec, dtype=np.float64)
        if (codes.ndim != 2 or codes.shape[0] != groups * ot or codes.dtype != np.uint16
                or not 16 <= codes.shape[1] <= 128 or codes.shape[1] % 2
                or rec.shape != tiles.shape or not np.isfinite(rec).all()):
            raise ValueError("encoder returned invalid packed/decoded tiles")
        if packed is None:
            packed = np.empty((groups, HAD // TILE, ot, codes.shape[1]), dtype=np.uint16)
        if codes.shape[1] != packed.shape[-1]:
            raise ValueError("encoder changed rate within a tensor")
        packed[:, start // TILE] = codes.reshape(groups, ot, -1)
        decoded[:, start:stop] = rec.reshape(groups, ot, TILE, TILE).transpose(0, 2, 1, 3).reshape(groups, TILE, cols)
    return packed.reshape(rows // TILE, ot, -1), decoded.reshape(rows, cols)


def allocate_bits(items, average_bpw=1.75):
    """Exact measured-loss multiple-choice budget for small pilot manifests.

    items: [{"weights": positive multiple of 256,
             "candidates": [{"n": even halfwords 16..128, "loss": nonnegative}, ...]}].
    Losses must share a measurement/normalization convention. This optimizes
    payload only, not model quality; scales/metadata are deliberately separate.
    # ponytail: sparse Pareto DP for pilot sets, use a scalable solver for full-model manifests.
    """
    if not items or not math.isfinite(average_bpw) or average_bpw <= 0:
        raise ValueError("need items and a positive finite budget")
    total_weights = 0
    choices = []
    for item in items:
        weights = item["weights"]
        if not isinstance(weights, int) or weights <= 0 or weights % 256:
            raise ValueError("weight counts must be positive multiples of 256")
        total_weights += weights
        candidates = []
        for candidate in item["candidates"]:
            n, loss = candidate["n"], candidate["loss"]
            if not isinstance(n, int) or n % 2 or not 16 <= n <= 128:
                raise ValueError("rate n must be even and in [16,128]")
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("loss must be finite and nonnegative")
            candidates.append((n, weights * n // 16, float(loss)))
        if not candidates:
            raise ValueError("each item needs measured candidates")
        choices.append(candidates)
    budget = math.floor(average_bpw * total_weights)
    frontier = {0: (0.0, ())}
    for candidates in choices:
        next_frontier = {}
        for used, (loss, path) in frontier.items():
            for n, bits, extra in candidates:
                cost, value = used + bits, loss + extra
                if cost <= budget and (cost not in next_frontier or value < next_frontier[cost][0]):
                    next_frontier[cost] = value, (*path, n)
        frontier, best = {}, math.inf
        for cost in sorted(next_frontier):
            if next_frontier[cost][0] < best:
                frontier[cost] = next_frontier[cost]
                best = next_frontier[cost][0]
        if not frontier:
            raise ValueError("budget cannot fit the candidate set")
    bits, (loss, rates) = min(frontier.items(), key=lambda x: (x[1][0], x[0]))
    return {"rates_n": list(rates), "payload_bits": bits, "weights": total_weights,
            "average_bpw": bits / total_weights, "loss": loss, "budget_bits": budget}
