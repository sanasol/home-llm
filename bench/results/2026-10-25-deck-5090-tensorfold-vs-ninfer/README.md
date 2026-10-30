# 2026-10-25 · Steam Deck + 5090 · TensorFold vs NInfer · Qwen3.8-27B

Stopgap host: Steam Deck OLED (Zen 2, 4C/8T, 14 GiB RAM) → M.2→OCuLink → Minisforum
DEG1 → RTX 5090 Astral. Link: **PCIe 3.0 x4**, BAR1 256 MiB, no ReBAR. Ubuntu 26.04,
kernel 7.0, driver 615.71 open, CUDA 13.4. Models read from microSD (~85 MiB/s).

Bench: [bench.py](bench.py), identical across engines. Greedy, thinking off, one stream.
- Short prompts (code / prose / list): 512 output tokens, median of 3 runs.
- Long prompts: Zig sources from TensorFold with a unique marker prepended (so the
  prompt cache never hits), 256 output tokens. Prefill = time to first token.

## Configs

| | Engine | Weights | Draft | Notes |
|---|---|---|---|---|
| **TF** | [BobClawblaw/qwen38-27b-vision-rtx-5090](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090) `2147ce9`, TensorFold `18d93451` | MLX 4-bit, 18.1 GiB | DFlash2 | `serve.sh`, KV fp8, 262K, `PROMPT_CACHE_GIB=4` |
| **NI-MTP** | [Neroued/ninfer](https://github.com/Neroued/ninfer) `64d10fbf` | NVFP4 artifact, 19.7 GiB | MTP, 3 tokens | README profile, `--host-context-mib 4096` |
| **NI-DF2** | same | 21.4 GiB with drafter | DFlash2, 7 tokens | same |

## Results (tok/s)

| | TF | NI-MTP | NI-DF2 |
|---|---|---|---|
| load from microSD | 307 s | 250 s | 270 s |
| short code | **410** | 203 | 257 |
| short prose | **241** | 162 | 163 |
| short list | **246** | 172 | 201 |
| 8.8K: prefill | 4,097 | **11,313** | 11,038 |
| 8.8K: decode | **230** | 193 | 222 |
| 35K: prefill | 3,872 | **10,856** | 10,633 |
| 35K: decode | **232** | 175 | 181 |
| 108K: prefill | 2,776 (38.8 s) | **7,315 (14.7 s)** | 7,270 |
| 108K: decode | 145 | 156 | **164** |

Draft acceptance on NInfer: MTP3 ~61% at 108K; DFlash2 29–45% (the drafter works
worse under NInfer than under TensorFold).

## Conclusions

- **Generation:** TensorFold + DFlash2 is clearly faster at short/mid context —
  ~1.6–2× on code.
- **Prefill:** NInfer is 2.6–2.8× faster. A cold 108K prompt: 15 s vs 39 s.
- At 108K NInfer decodes slightly faster (156–164 vs 145).
- The narrow Deck link barely costs anything: TF reproduces the author's full-PC
  README numbers (35K: 3,872 / 232 vs 4,432 / 196).
- For agent workloads it's a tradeoff: short turns with cache reuse → TF wins;
  frequent fresh large context → NInfer wins. NInfer also ships an Anthropic
  Messages API.

Raw per-run JSONs: `bench-tensorfold.json`, `bench-ninfer-mtp3.json`, `bench-ninfer-dflash2.json`.
