"""Calibrated SPEC1 trunk integration; never reads/substitutes official weights.

CPU-only checks: python3 -m unittest discover -s ds-sushi-abl -p test_spec1_quality.py

Integration: OrcaReader returns dense [out,in]. Transpose wo_b explicitly.
For grouped wo_a reshape donor [groups*out,in] to [groups,out,in] THEN
transpose(0,2,1). Supply matching wo_a_in[group] Grams, never a pooled Gram.
encode_trunk returns .trellis/.suh/.svh arrays ready for the trunk writer; caller
owns config stamps and provenance. Each group uses only its passed donor values.
"""
import hashlib
import numpy as np
import stq_quality as quality


def _rate(n):
    if not isinstance(n, int) or n % 16 or not 16 <= n <= 112:
        raise ValueError("stq SPEC1 search supports integer K1..K7 (n16..112) only")


def stq_encoder(n, device="cpu", batch=1):
    """Lazy adapter to existing f32 SPEC1-w15 search; no implicit CUDA use.

    Search input is trellis order, decoded callback output is row-major. This
    adapter's f32 least-squares is not Sushi float16-cost byte parity.
    """
    _rate(n)
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    import torch
    import stq

    def encode(tiles):
        x = torch.from_numpy(np.ascontiguousarray(tiles.reshape(-1, 256)[:, stq.PERM], dtype=np.float32)).to(device)
        codes, _ = stq.encode_tiles(x, n, batch=batch)
        codes = np.asarray(codes, dtype=np.uint16)
        values = stq.spec1_table(n)[stq.unpack_codes(codes) & ((1 << stq.W_BITS) - 1)]
        decoded = np.empty_like(values)
        decoded[:, stq.PERM] = values
        return codes, decoded.reshape(-1, 16, 16)

    return encode


def _stored_scale(values):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = np.asarray(values).astype(np.float16)
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("scales overflow/underflow stored F16; choose another gain")
    return scale


def encode_matrix(donor, hessian, n=96, *, seed=0, name="trunk", gains=(1.0,),
                  encode=None, device="cpu", batch=1):
    """Encode edited donor [in,out], calibration-select gain by public Gram loss.

    Gains use actual rounded-F16 scales for BOTH target and Hessian. Selection
    is calibration-only; held-out/model evaluation remains the caller's job.
    encode can inject a row-major tile codec, otherwise uses stq_encoder.
    Returns trellis, suh, svh, calibration_loss, gain and source weight digest.
    """
    _rate(n)
    donor = quality._matrix(donor, "donor")
    hessian = np.asarray(hessian)
    diagonal = hessian.ndim == 1
    hessian = (quality._diagonal(hessian, donor.shape[0]) if diagonal
               else quality._hessian(hessian, donor.shape[0]))
    if any(d % 128 for d in donor.shape):
        raise ValueError("trunk dimensions must be multiples of H128")
    gains = tuple(gains)
    if not gains or any(not np.isfinite(g) or g <= 0 for g in gains):
        raise ValueError("gains must be positive and finite")
    if not isinstance(name, str):
        raise ValueError("tensor name must be a string")
    # Stable local RNG; independent tensor/group names must not reuse sign streams.
    identity = f"{seed}:{name}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))
    signs = [rng.choice([-1, 1], size=d) for d in donor.shape]
    _, raw_su, raw_sv = quality.regularize(donor, *signs)
    sv = _stored_scale(raw_sv)
    encode = stq_encoder(n, device=device, batch=batch) if encode is None else encode
    best = None
    for gain in gains:
        su = _stored_scale(raw_su / gain)
        inner = quality.target_inner(donor, su, sv)
        if diagonal:
            l = quality.prepare_diagonal_ldl(hessian, su)
            codes, decoded = quality.ldlq_blocks(inner, l, encode)
        else:
            l, _, _ = quality.prepare_ldl(quality.rotated_hessian(hessian, su))
            codes, decoded = quality.ldlq(inner, l, encode)
        if codes.shape[-1] != n:
            raise ValueError("codec returned a rate different from requested n")
        public = quality.reconstruct(decoded, su, sv)
        loss = (quality.weighted_error_diagonal(donor, public, hessian) if diagonal
                else quality.weighted_error(donor, public, hessian))
        if best is None or loss < best["calibration_loss"]:
            best = dict(trellis=codes, suh=su, svh=sv, calibration_loss=loss,
                        gain=float(gain), n=n, name=name,
                        donor_sha256=hashlib.sha256(np.ascontiguousarray(donor, dtype='<f8').tobytes()).hexdigest())
    return best


def encode_trunk(donor, hessian, n=96, **kwargs):
    """Encode [in,out] wo_b or [group,in,out] wo_a, returning stacked pack arrays.

    Grouped Hessians must be [group,in,in] or native diagonals [group,in].
    Ordinary matrices also accept a native diagonal [in]. No original-bank arguments exist:
    fresh signs, scales and codes are produced from the edited donor alone.
    No guessed GGUF reshape: caller supplies the explicit public orientation.
    """
    donor, hessian = np.asarray(donor), np.asarray(hessian)
    if donor.ndim == 2:
        return encode_matrix(donor, hessian, n, **kwargs)
    if (donor.ndim != 3 or donor.shape[0] == 0
            or hessian.shape not in ((donor.shape[0], donor.shape[1], donor.shape[1]),
                                     (donor.shape[0], donor.shape[1]))):
        raise ValueError("grouped donor needs [group,in,out] and Hessian [group,in,in] or diagonal [group,in]")
    name = kwargs.pop("name", "trunk")
    groups = [encode_matrix(w, h, n, name=f"{name}.group{g}", **kwargs)
              for g, (w, h) in enumerate(zip(donor, hessian))]
    return {"trellis": np.stack([g["trellis"] for g in groups]),
            "suh": np.stack([g["suh"] for g in groups]),
            "svh": np.stack([g["svh"] for g in groups]), "groups": groups,
            "calibration_loss": sum(g["calibration_loss"] for g in groups), "n": n}


def allocate_tied_experts(items, average_bpw=1.75):
    """Measured-loss pilot allocation with one shared gate/up/down rate per expert.

    Sum projection weight counts/losses in each item before calling. PAIR1's
    21 possible even rates n24..64 guarantee <=21 joint signatures, hence <=32
    consumer groups. General independent projection rates need a constrained
    allocator, not truncating the exporter's rate groups after optimization.
    Exporter owns the exact common expert/router permutation, not this helper.
    """
    for item in items:
        if any(c["n"] % 2 or not 24 <= c["n"] <= 64 for c in item["candidates"]):
            raise ValueError("tied expert candidates must use PAIR1 even n24..64")
    result = quality.allocate_bits(items, average_bpw)
    result["group_signatures"] = [tuple([n] * 3) for n in sorted(set(result["rates_n"]))]
    result["rate_plan"] = {str(i): dict(gate=n, up=n, down=n) for i, n in enumerate(result["rates_n"])}
    return result
