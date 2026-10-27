# Community GPU-inference repos — transferability assessment (2026-10-27)

Research on 4 community repos for Qwen3.8-27B on our rig (2× 3090 no-P2P + 5090 Strata main).
All facts below are from the repos' own READMEs/docs/bench cards (shallow clones, read 2026-10-27).
Repos cloned at: `~/.hermes/cache/scratch/repo-research/`.

## 1. Sandermage/sndr_core_engine ("Genesis") — **SKIP**

- **Stack:** patch-overlay (git-apply `--3way` of 99 patches in `patches/series` over a clean vLLM clone), SNDR v12.1.0. Pin: **vLLM 0.23.1rc1.dev748**, Python 3.10–3.13, CUDA 12.x, FlashInfer 0.5.0. Overlay only — never bundles a vLLM version.
- **Models:** Qwen3.6 (7B/27B/35B-A3B) + Gemma-4. **No Qwen3.8 anywhere in the catalog** (only in archived files/test scripts). Single-3090 27B preset `qa-qwen3.6-27b-tq-1x` = 78K ctx.
- **Spec decode:** MTP (27B TQ-k8v4: K=4; K=5 "over-proposes bad structural tokens → unparseable tool-calls"; TQ×MTP needs `GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY=1` or it degenerates into token repetition) + suffix decoding (P75) + ngram. 27B TQ accept: 0.70-0.73 (K=3 era).
- **KV:** TurboQuant K8V4 (8-bit K / 4-bit V, calibrated per checkpoint, block 64).
- **Numbers (2× A5000 reference rig, no NVLink):** 27B int4 TQ single-stream 127.4 t/s (K=5, TPOT 7.54 ms); 35B FP8 239.7 (K=5); multi-conc-8 aggregate 471 (27B) / 672 (35B).
- **NVLink:** "not critical for 27B TP=2 — PCIe Gen4 x16 sufficient, all-reduce ~3-5% of step time". Dual = PCIe P2P, no NVLink.
- **Maintenance:** last commit 2026-07-08 (~3.5 months stale as of now); README claims weekly releases.

**Verdict: SKIP.** No Qwen3.8 support, vLLM pin 8 minor versions behind our 0.31/0.32 stack, 78K single-card ceiling, A5000-calibrated. Useful only as a pattern reference (how to structure a version-tolerant patch overlay; their #40880 hybrid-KV fix was later upstreamed as vllm#40914).

## 2. halt95/merlin-engine (Merlin 3.0.0) — **ADOPT-CANDIDATE** (for single-3090 262K)

- **Stack:** **full fork of vLLM 0.32.0** (upstream 0.32.0 RC + 5 next-RC commits + final release commit a14676c711 + one macOS-only fix #62177; 11 earlier backports superseded by upstream). Ships as a prebuilt wheel `vllm-0.32.0+merlin.7002f56537-cp313` (sm_80/86/89), Python 3.13, CUDA 13, FlashInfer 0.7.0.post1 (32-row verify-tile patch). Not an overlay — a drop-in vLLM replacement.
- **Qwen3.8-27B recipe** (`models/qwen38-27b/`): philbert AutoRound W4A16 int4 body + int4 lm_head + int4 MTP drafter + **calibrated FP8 E4M3 KV** (`kv_scales-philbert-e4m3.json`) + fp16 SSM state; **RecoverSSM** (`--use-replayssm --mamba-cache-mode align`) for GDN/Mamba-2 state recovery under MTP verify; **MTP K=4** (bumped from K=3 2026-09-20: +3.7–8% wall, 262K pool 794K→826K); prefix match unit 16; 960-token hybrid KV blocks at K=4 (17 attention layers × 960 × 2048 B = 33,423,360 B/block).
- **1× 3090 (TP1) profile — the direct comparison to our B:**
  - decode **124 / 95 / 73 t/s** at 4K / 131K / 262K (thinking on, medium); exactly 262K thinking off: **49.3**; prefill 456 t/s @262K (574 s TTFT).
  - KV pin 9,692,774,400 B → **pool 271,506 tokens = exactly one 262K session** (or two 131K); max 2 sequences; 4 MP images (vision tower in host RAM).
  - GSM8K-200 194/200; 262K needle exact; **prefix follow-on reuse 98.2–99.8%, follow-on prefill 1401–1517 t/s** (124K needle: 1715 t/s) — the one vLLM 3090 stack in this research with working prefix caching.
  - **No P2P needed (single card).**
- **2× 3090 (TP2) profile — requires P2P:**
  - decode 207 / 173 / 142 (4K/131K/262K thinking on); 111.1 @262K thinking off; ITL 19.99 ms @4K; 4.06–4.57 tokens/event (MTP K=4); **3× 262K resident (pool 826,689 = 883 blocks/card), 0 preemptions**; 8 agents: 517 (thinking on) / 735 (off); 107.5/229.1 narrative/code @4K.
  - Launcher sets `VLLM_SKIP_P2P_CHECK=1 NCCL_P2P_LEVEL=SYS` — reference host runs **P2P via the open-kernel-module** (aikitoria's patched NVIDIA open-source kernel module, i.e. P2P enabled on consumer Ampere — this is what made TP=2 work, NOT NVLink).
  - `27b-tp2-3090-nop2p` profile exists but is **untested** ("no non-P2P host has been run"). On no-P2P hosts they recommend **N × TP1 single-card servers**: measured 4× CMP 170HX (no P2P), 8 agents × 16K: **TP1×4 = 873 tok/s vs TP2×2 = 523 vs TP4 = 371**. On P2P hosts, decode-sized allreduce uses FlashInfer's PCIe IPC (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) + NCCL Simple for prefill + `--disable-custom-all-reduce`.
  - 4× 3090 (TP4): 250/235/215; 7× 262K; 8 agents 752/1036.
- **Requirements:** Linux x86_64, driver R580+, Python 3.13, uv, gcc13, ~1 GiB /dev/shm; 2.4 GiB pinned host RAM (embeddings).
- **Maintenance:** very active — 3.0.0 released 2026-10-20/21, commits to 2026-10-22.

**Verdict: ADOPT-CANDIDATE** for the *single-262K-session* profile on one of our 3090s (same card, same model family, same vLLM generation as HyperQwen): 73–124 t/s vs our B's 22–87, plus working prefix cache (98–99.8% follow-on) which our vLLM 3090s lack. Cost: it's a 0.32.0 **fork** — swapping an engine, not patching HyperQwen; and its 1× 3090 pool holds **one** 262K session (our KVarN B holds 2–3). TP2 on our no-P2P host = their untested profile, and their own numbers say 2×TP1 beats TP2 on no-P2P — so TP2 is out unless we adopt the open-kernel-module driver (aikitoria) to enable P2P.

## 3. r0b0tlab/qwen38-exl3-dflash2 — **TEST-WORTHY** (DFlash2 reference; engine not portable)

- **Stack:** ExLlamaV3 community fork (`r0b0tlab/exllamav3` @ 355c6ee, v1.5.0+) + torch 2.10.0+cu130 + CUDA 13. **Not vLLM** — no transfer to HyperQwen as code.
- **Model:** Qwen3.8-27B EXL3 4-bit (12.4 GiB) + **DFlash2 draft model 1.2 GiB** (`incoai/Qwen3.8-27B-DFlash2`, llama.cpp port, Apache-2.0).
- **What DFlash2 actually is:** a **separate small draft network** (NOT a GGUF draft of the same weights, NOT the target's own MTP heads):
  - **block-diffusion drafter**: appends `block_size-1` (7) mask tokens, ONE bidirectional draft forward (`causal=False`) proposes a whole block (block_size 8);
  - builds its own draft KV from **taps on the target's hidden states at layers [5,19,33,47,61]** (fc → norm → per-draft-layer k/v proj + rope);
  - **GroupedDynamicCausalConv** (kernel 2, group 16) wrapped around each draft layer's attention and MLP sublayers;
  - **CandidateSelector**: predecessor/successor codebooks (rank 256) over top-K=16 of the *target's* LM-head logits, walks the block sequentially;
  - verify: one target forward over [last + 7]; at T>0 uses **q-aware rejection sampling** (accept iff u < p_i/q_i; on reject, bonus token from normalized (p−q)+ residual) → **lossless** w.r.t. the sampling distribution; greedy path is plain token-match.
- **KV:** bf16 full-attention (16 layers) + GDN recurrent state 1.22 GiB/seq-slot (fp32, 48 layers × 8 rows). 262K fits **1 sequence** (11.68 GiB); 4 concurrent @1024 = 19.5 GiB.
- **Numbers (1× 3090, 1 stream, GSM8K-8K):** DFlash2 **162.9 t/s** (accept length 5.66, 9.7 tok/step) vs MTP 116.3 (4.12) vs AR 42.8 (1.00). **Accept degrades with depth:** 150K → 1.82 (25.3 t/s), 225K → 1.82 (13.7), 260K → 1.83 (9.7). Prefill 262K = 341 s TTFT.
- **Maintenance:** last commit 2026-09-20; project reads as concluded (result adjudication).

**Verdict: TEST-WORTHY, but the test is cheap:** DFlash2 is **already in our HyperQwen stack** (setup B: DFlash2 W4A16 drafter, 7 tokens per verify = the same block-7 window). No engine port needed — the open question is just which of A's setups (MTP-fp8-150K vs DFlash2-bf16-64K vs DFlash2×15/CTX=long) wins on our workload mix. Caveats straight from club-3090's HyperQwen page: on sm_86 HyperQwen's DFlash2 path has **no fp8 KV** (FLASH_ATTN refuses fp8 there) → 64K bf16 (or int8 per-token-head at 128K via `CTX=long`), and DFlash2 accept falls off at long context (per this repo, ~1.8 at 150K+).

## 4. noonghunna/qwen36-dual-3090 → **noonghunna/club-3090** — **TEST-WORTHY** (reference; our engine's public bench data)

- **Moved** 2026-04-28 to `noonghunna/club-3090` (v0.12.0, active). The old repo: dual-3090 vLLM TP=2, MTP, fp8 KV, Qwen3.6-27B.
- **NVLink? No — explicitly.** "All numbers in this repo are measured PCIe-only — no NVLink bridge… NVLink bridge NOT required." What made TP=2 work on consumer Ampere:
  1. `--disable-custom-all-reduce` (vLLM's custom kernel assumes NVLink-class bandwidth → fall back to NCCL);
  2. `NCCL_P2P_DISABLE=1` (host bounce buffer; "counterintuitively faster on PCIe-only");
  3. `NCCL_CUMEM_ENABLE=0` (NCCL cumem pool compatibility issues);
  4. **the Marlin pad fix — vllm#40361** (`GPTQ_MARLIN_MIN_THREAD_N=64` blocks per-rank out-dim <64 shards of AutoRound INT4 under TP=2 on sm_86). club-3090 vendored it until **upstream #45295 + #45176** landed (native in vLLM v0.24.0+).
  - Their measured result: dual-3090 TP=2 MTP fp8 **69.7–70 / 89–92 t/s @ 262K** (2× 3090, 230 W, no P2P, custom-AR off, MTP n=3 accept 2.86–4.0) — i.e. TP2 on no-P2P 3090s works but loses to 2×TP1 (131/207 @180K). A community rig with **P2P + custom-AR on measured +15.7%** over the no-P2P row (club-3090 discussion #773) — the quantified price of our no-P2P condition.
  - This is consistent with our earlier TP=2 attempt dying at 6–13 t/s being a *recipe* problem (Marlin sub-tile / GDN / hybrid-KV bugs), not a fundamental interconnect wall.
- **Club-3090 extras relevant to us:**
  - **Our engine (HyperQwen) benchmarked on exactly a 3090:** slug `vllm-hyperqwen/qwen38-27b-single-ultrafast` (syv-ai image, vLLM **0.31.0 + 50 patches**, DFlash2 n=7, **bf16 KV, 64K**): **131.4 / 230.1 decode** (narrative/code), CV 3.6/4.3%, TTFT 87 ms, **DFlash2 mean accept 4.21**, 23,292 MiB, 3090 stock 370 W, 2026-10-14 (@noonghunna); on 5090: 251.3 / 415.5, accept 3.93, 8-pack think 127/150. → On a 3090, HyperQwen-DFlash2@64K beats our current A (MTP fp8 150K: 90–105) on raw decode, at half the context; DFlash2 fp8 KV is impossible on sm_86 (FLASH_ATTN refuses).
  - HyperQwen's 50-patch series (per their page): int8 Marlin GEMM layer selection, MTP draft vocabulary, **GDN spec-decode bounds (vllm#50021)**, hybrid-KV CUDA-graph fix, KVarN 4/2-bit KV cache; setups B (DFlash2 n=7, bf16 64K), C (n=15), D (MTP, int8 KV, 150K), E (KVarN, 240K), batch, and 2-card via `EXTRA_ARGS=--tensor-parallel-size 2`.
  - **W4A8 toggle (env-only): prefill +50% (1,674→2,513 t/s), decode neutral** on the v0.25 dual stack (#609) — candidate micro-optimization for prefill on our vLLM boxes.
  - Caveats on the HyperQwen slug: its own hardened chat template (system message must lead; tool-schema key order affects prefix cache), no thinking budgets wired, only setup B catalogued, K8/V4 KVarN not in syv-ai's image.

**Verdict: TEST-WORTHY** — nothing to port into HyperQwen itself, but it is (a) the definitive no-NVLink TP2 recipe for 3090s (the three NCCL flags + Marlin fix, now upstream) and (b) the only public bench of *our exact engine on our exact card*: DFlash2 131/230 @64K.

## Bottom line for OUR rig

| Candidate | What it gives | What it costs |
|---|---|---|
| **Merlin 1× 3090 profile** (repo 2) | 262K on one 3090, 73–124 t/s, **prefix cache 98–99.8% working**, MTP K=4 + RecoverSSM, calibrated fp8 KV | Swap B's engine to a 0.32.0 fork; pool = **1** 262K session (KVarN B holds 2–3); 3.5-month-old vs our stack's cadence risk; 574 s TTFT on 262K prefill |
| **HyperQwen setup B (DFlash2)** (repos 3+4) | 131/230 @64K on a 3090, accept 4.21 — already in our image | 64K bf16 only on sm_86 (no fp8 KV for the DFlash2 path); accept ~1.8 at 150K+ |
| **TP2 on our no-P2P pair** | +15% decode over no-P2P baseline if we enable P2P (open-kernel-module, aikitoria) + custom-AR | Driver swap on the box that runs the gateway; club-3090's no-P2P dual = 68/92 @262K still < 2×TP1 |
| **sndr_core_engine** (repo 1) | — | No Qwen3.8, old pin → **SKIP** |

Recommended order of experiments (all on idle slots, no live-endpoint benches without a go):
1. A → HyperQwen setup B (DFlash2, 64K): measure vs current MTP at 8K/40K/90K; decide by workload ctx distribution.
2. B → Merlin 3.0.0 1× 3090 (one 262K session, fp8 KV, MTP K=4 + RecoverSSM, prefix cache): measure vs current KVarN at 245K single-stream; keep KVarN if the 2–3-session fan-out is the actual job.
3. Only if both of the above disappoint: consider the open-kernel-module P2P path for TP2.

---

## Experiment 1 executed 27.10.2026: DFlash2 on A (HyperQwen, rig 3090 #1 OCuLink)

Setup: drop-in on `hyperqwen-a.service`: `SPEC=dflash2 DRAFT=models/Qwen3.8-27B-DFlash2 FA2_FP8KV=1` (the
sm86 default arm; required building the FA2 fp8-KV plugin via `fa2/install.sh`, ~2 min, nvcc 13 from pip).
Memory geometry: the single-user launcher pins `KV_MEM` (default 5.2 GiB, skips profiling, ignores
gpu_memory_utilization). The 3.6 GB full-model DFlash2 drafter does not fit at 131K context on 24 GB ->
OOM at first step. Working arm: `KV_MEM=4100000000 MAX_LEN=96000` (KV 96,680 tokens, 1.01x at 96K).
Note: the unit's default `TimeoutStartUSec=90s` kills every cold start; add `TimeoutStartSec=900` in the drop-in.

Decode t/s, identical Russian prompts, temp 0, 300 out, per-request counters (MTP = production arm, 150K ctx;
DFlash2 arm capped at 96K):

| ctx | MTP (2 runs) | DFlash2 (3 runs) | delta |
|---|---|---|---|
| 8K  | 124.7 / 107.5 | 132.6 / 131.1 / 108.4 | ~+5%, noisy |
| 40K | 96.3 / 115.3  | 82.3 / 90.7 / 86.6    | **-13%** |
| 90K | 93.2 / 100.8  | 107.3 / 110.5 / 107.3 | **+11%** |

Acceptance: 2438/5551 draft tokens (~3.07 accepted per draft, k=7).

Verdict: **not adopted** — A reverted to production MTP/150K arm. DFlash2 wins only at long context
(+11% @90K) and loses mid-context; it also cuts the context ceiling 131-150K -> 96K because the drafter
eats the KV budget. Keep as an optional profile (drop-in recipe above) for long-ctx-only sprints.
Next per plan: Merlin 1x3090 profile test on B.

