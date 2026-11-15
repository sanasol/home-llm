# Public journal

Dated build log of the home LLM rig. Every entry points at the harness/config and the raw
numbers in this repo; no private hosts, keys or personal data. Hardware line: Ryzen 9 9950X,
96 GB DDR5-6000, RTX 5090 (PCIe 5.0 x8) + 2x RTX 3090 (external PCIe 3.0 x4 OCuLink / internal PCIe 4.0 x8; external Gen3 since 13.11), Ubuntu 26.10,
driver 615.71, CUDA 13.1.

### 10-11.10.2026 — Mac M3 Max 48 GB, phase 0
- Qwen3.8-27B 4-bit, DFlash2 speculative decode; 128K context inside a 34 GiB budget.
- Long-chat bench, 5 engines, same 20K-token agent chat: Splash+DFlash2 **33.4 t/s**, TensorFold
  26.7, oMLX 23-26, llama.cpp MTP 12.7, MTPLX swapped at 107K.
- Lesson: prefill, not decode, decides agent latency. Harness + raw JSON: `mac-local/bench/`.

### 25-26.10.2026 — Steam Deck + DEG1 + RTX 5090 (PCIe 3.0 x4), phase 1
- NInfer serving Qwen3.8-27B NVFP4 (21.7 GiB weights), 2 streams, vision, 240K/request.
- TensorFold decode up to **410 t/s** on short code, but 2.6-2.8x slower prefill; NInfer
  **11.3k t/s prefill**, ~257 t/s code decode. Winner for agent traffic: NInfer.
- KV cache fp8 -> **k8v4**: KV pool +28%, 240K fits with vision, no speed loss; DFlash2 window
  K=7 is the optimum. The narrow OCuLink x4 link barely costs decode.
- Raw results: `bench/results/2026-10-25-*`, `2026-10-26-*`.

### 26.10.2026 — 3-GPU rig assembled
- Ubuntu 26.10 headless over Wi-Fi, rescue ISO; services run from systemd units in `rig/config/`.
- 2x RTX 3090 without P2P: vLLM TP=2 **6.5 t/s** (all-reduce through host SHM), PP=2 **44.6 t/s**,
  then one engine per card **95-100 t/s**. TP on GeForce without NVLink is a trap.

### 26-27.10.2026 — thermals, monitor, observability
- Blower 3090 fan floor **41% -> 0% idle** via a driver runtime override; ~45% to hold 78C
  under load. Ampere Cert 3.0 prevents flashing a modded VBIOS; the override was the way in.
- Web monitor :8090 (SQLite history), LiteLLM single endpoint, Prometheus/Grafana, Telegram
  alerts, watchdog outside the box.
- GPU-unlock: BF16 GEMM **103.4 -> 66.5 TFLOPS (+55%)**, decode/prefill unchanged.

### 27-29.10.2026 — engines, quantization, sampling
- vLLM 0.32 27B GPTQ TP=2 without P2P: 6.5-13 t/s; PP=2 44.6 t/s. Lean GPTQ int4 path hit
  GPTQModel's lm_head auto-bump -> 40 vs 160 t/s; DP=2 128K did not fit (max ~85K).
- HyperQwen (vLLM 0.31 + patches) fp8 + fast pack: RU 95, EN 104-105 t/s, 150K context;
  Cyrillic draft vocab for MTP: RU 55 -> **101.5 t/s**; upstream PR #423 merged.
- OpenJet trial: synthetic bench claimed 2.6x, real agent task **531 s vs 116 s** -> rollback.
- Russian glitches in Qwen3.8 long answers: 18.3% -> **1.7%** with T 0.7 + min_p 0.05
  (scanner + A/B harness in `rig/scripts/`).

### 30.10-01.11.2026 — Flash-Next on all three cards
- Qwen3.8-Flash-Next IQ3_S on 5090 + 2x 3090, one shared 50 GB expert arena in RAM tmpfs:
  **371-392 t/s aggregate** for 4 streams, 512K context on the 5090 (YaRN x2, KV q4_0).
- Gotcha: `/dev/shm` per-user quota kills a 50 GB arena silently at ~36 GiB -> dedicated tmpfs.
- Cyrillic draft vocab in Strata: RU **151 -> 212 t/s**, acceptance 42% -> 69%.
- Abliterated model on all lanes; conversation parking: 102K-token return **0.9 s** (vs 15 s),
  442K-token return **4.8 s** (vs 74 s cold).
- RGB equalizer phases decode/prefill (OpenRGB).

### 06.11.2026 — session save bug
- Strata 0.2.11's `drafting: auto` flipped to DFlash2 and silently broke session-file saves
  (`invalid K/V extent`); wrote the guard (MTP draft state passed only when present) that became
  upstream PR #1492 and shipped in 0.1.57.2.

### 10-12.11.2026 — DFlash2 rounds, layer-major, Maya
- DFlash2 on cards that stream experts from RAM: RU 119 -> 107 t/s (-10%), acceptance formally
  higher (0.79 vs 0.68) but offered-per-token lower; tuned pack + spec 8 still worse. MTP +
  Cyrillic vocab stays fastest for Russian. Layer-major prefill pays only for drive-fed experts;
  with RAM-resident experts it either refuses to engage or collapses to ~10 t/s.
- DeepSeek-V4.1-Flash (Maya) on the whole box: DSpark decode **48 -> 68 t/s** (+41%, accept 77%),
  cold prefill 1429 -> 2690 t/s (x1.9) with three opt-in flow flags, 483K tokens in 7.5 min.
- Layer split tuned to link speed (5090 28.9 GB/s, 3090 x8 13.4, 3090 x4 6.7): split `16,8,16`
  after round 2; a 14,9,17 split reads cold 16% faster but agent turns 15% slower.

### 12.11.2026 — upstream work
- Reported a 5090 + 2x 3090 Maya box (PR #959 prefill-on-CPU), three Strata comments
  (disk conversation cache, DFlash2 + Cyrillic on NVIDIA streaming cards, select-F16 topology).

### 13-15.11.2026 - Gen3 mitigation, Kiwi and overnight production
- Repeated Xid 79 events on the external Gen4 OCuLink path required a dock power-cycle. BIOS Gen3 x4 reduced receiver errors; 90 s / 304 GB transfer test measured 3.38 GB/s. The exact faulty component is not isolated.
- Current pack: Kiwi Maya-XS abliterated. Local Maya v1.1.61 + PR #959 deltas + PR #911, engine `56154a5`, CPU IQ1_S VNNI enabled. 512K, DSpark, parallel 2 and split 16,8,16 retained; runtime refusal-direction ablation off by default. Experimental, not a proven general speed/quality win.
- [15 November overnight samples](../bench/results/2026-11-15-kiwi-overnight/): 00:00-08:30 CET, 69.0 t/s median non-zero sampled live decode gauge, 258,665 completed output tokens. Ordinary mostly-idle traffic, not a controlled benchmark. Older XS v2 cold-prefill measurements remain historical.
