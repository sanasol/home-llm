"""Optional torch CUDA Engram FP8 quantization; no custom kernel or GPU IPC.

Same F32 amax floor/power-of-two scale/E4M3FN RNE as engram_export reference.
Importing this module does not initialize CUDA. Caller owns GPU scheduling.
"""
import numpy as np


def quantize_rows(rows, device='cuda', metrics=False):
    import torch
    x = np.asarray(rows,dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % 32 or not np.isfinite(x).all():
        raise ValueError('Expected finite F32 [rows,columns] with columns multiple32')
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    blocks = t.reshape(x.shape[0],-1,32)
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    ratio = amax * np.float32(1/448).item()
    mantissa, exponent = torch.frexp(ratio)
    power = exponent - (mantissa == .5).to(exponent.dtype)
    if bool(((power < -126) | (power > 127)).any()):
        raise ValueError('Scale outside normal finite E8M0/F32 range')
    scales = torch.ldexp(torch.ones_like(amax),power)
    normalized = blocks / scales[...,None]
    if bool((normalized.abs()>448).any()):
        raise ValueError('Scale calculation overflow')
    fp8 = normalized.to(torch.float8_e4m3fn)
    codes = fp8.view(torch.uint8).reshape(x.shape).cpu().numpy()
    stored_scales = (power+127).to(torch.uint8).cpu().numpy()
    if not metrics:
        return codes,stored_scales
    # Native FP8*power2 is exactly representable in F32 for accepted exponents;
    # F64 error/signal reduction avoids large host dequant/metric temporaries.
    decoded = fp8.float()*scales[...,None]
    error = blocks.double()-decoded.double()
    sse = float((error*error).sum().item())
    signal = float((blocks.double()*blocks.double()).sum().item())
    return codes,stored_scales,{'sse':sse,'signal_sq':signal}
