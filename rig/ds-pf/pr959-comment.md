Tested 5ea06e1 on a third, quite different box: one consumer CPU (AVX-512), three GPUs on a layer split with mixed PCIe links. Only this PR's setting differs between the arms.

**Box:** Ryzen 9 9950X (16C/32T, AVX-512, one socket), 96 GB DDR5-6000 (dual channel), Linux, driver 615.71, CUDA 13.1. GPUs: RTX 5090 32 GB (PCIe 5.0 x8, the engine's ring line: 28.9 GB/s), RTX 3090 24 GB on PCIe 4.0 x4 (OCuLink, 6.7 GB/s), RTX 3090 24 GB on PCIe 4.0 x8 (13.4 GB/s).

**Model / engine:** DeepSeek-V4.1-Flash-1017 Maya-XS (v2), v1.1.61 (`c5842c5`) + this PR cherry-picked, `--tp 2` (falls back to a layer split, one GPU a group) with `DS4_SPLIT=18,9,13` (fewer layers on the x4 card), `--parallel 2`, `--max-context 524288`, DSpark on, `DS4_RING_MB=400`, 16 CPU threads. 9,250 experts in VRAM, 6,578 (47 GB) copied into RAM and pinned.

**Harness** (as in the PR description): the same binary in both arms, only `MAYA_DS4_PREFILL_CPU=0/1` differs; a fresh engine per run, one frozen `expert-profile.bin` restored before every start, `MAYA_DS4_SLOT_KEEP=0`, `MAYA_DS4_PROMPT_PICK=length:96`, `DS4_PREFILL_STATS=1` in both; order off / on / off / on, the same requests every run. Per run: two warm-up requests, three fresh prompts (nothing reusable, first token only), a ~24.5K-token start read once, then six appends of 590-3,840 new tokens on it, three fixed prompts for the first token's top-10 log-probabilities, three 300-token greedy answers. Times are the engine's `prompt N = R reused + X read in T ms` lines.

| prompt | without (`=0`), s | with (`=1`), s | change |
| --- | ---: | ---: | ---: |
| 396 tokens, fresh | 2.01 / 1.86 | 1.21 / 1.36 | -34% |
| 700 tokens, fresh | 2.51 / 2.56 | 1.71 / 1.66 | -34% |
| 1,984 tokens, fresh | 3.52 / 3.61 | 3.02 / 2.87 | -17% |
| 24,453-token start, cold | 17.03 / 16.86 | 13.92 / 13.95 | -18% |
| agent turn: 587 new tokens | 2.12 / 2.16 | 1.51 / 1.51 | -29% |
| agent turn: 681 new tokens | 2.46 / 2.46 | 1.86 / 1.86 | -25% |
| agent turn: 850 new tokens | 2.56 / 2.61 | 2.01 / 1.97 | -23% |
| agent turn: 1,344 new tokens | 3.06 / 3.06 | 2.36 / 2.36 | -23% |
| agent turn: 3,780 new tokens | 4.38 / 4.41 | 3.96 / 3.95 | -10% |
| agent turn: 3,838 new tokens | 4.36 / 4.48 | 3.99 / 4.01 | -10% |

Two runs per arm, each value one run; the two runs of an arm agree within 0.15 s.

- **The 24.5K cold start is 18% faster too.** In the PR description a 20.6K cold read was unchanged ("chunks of 8,192: no share"). Here the prompt path reads in chunks of 4,096 (`DS4_PREFILL ... chunk=4096` on every read of this run, the auto chunk for this split and VRAM), which is exactly `MAYA_DS4_PREFILL_CPU_MAX_T`, so every chunk of the cold read gets a share (one GPU's line for that read: `layers=78 experts=2746 rows=12017 cpu_ms=2657 wait_ms=3078`).
- **Decode, 300-token answers** (tok/s, every answer): without 55.8 / 79.8 / 70.0 / 60.3 / 77.4 / 68.6, with 57.4 / 77.6 / 67.3 / 57.2 / 77.8 / 68.5 - means 68.7 / 67.6, the same within the spread.
- **Where the share goes** (`DS4_PREFILL_STATS`, one 3.8K append): the 5090 group (18 layers) took 953-1,091 experts / 2,475-3,576 rows (cpu_ms 657-754), the x8 3090 570-589 experts (cpu_ms 251-264), the x4 3090 only 108-115 experts (cpu_ms 41-49). `wait_ms` was 0-29 ms except one 360 ms on the 5090 group (the CPU the limit there), `lane_busy=0` on the first runs. With the default `ROW_MS 0.49` the x4 card, the slowest link, gets the smallest share - the planner weighs the stream by `EXPERT_MS` alone, not by the GPU's link. A per-GPU expert time (the ring line already measures each link) would probably shift more experts off the x4 card.
- **Answers:** first-token top-10, KL(without ‖ with) over the common tokens: 1,759 tokens 2.2e-2 to 6.4e-2 (without vs without, across a restart: 1.2e-2); 744 tokens 3.3e-2 to 8.2e-2 (without vs without 8.1e-2); 377 tokens 4.3e-2 to 4.0e-1 (without vs without 1.6e-1). The first token was the same in every run at 744 and 1,759 tokens; at 377 tokens "English" and "The" sit 0.1-0.5 nats apart and swapped between restarts in both arms, not only with the setting. So the drift is the same size as this engine against itself across a restart on this box.

Not tested: `MAYA_DS4_PREFILL_CPU_EXPERT_MS` / `ROW_MS` sweeps, `DS4_FLOW_COPY_DMA` together with it (separately: with `DS4_FLOW_COPY_DMA=1 DS4_AI_RUN=1 DS4_YOCO_FLOW=1` on top, agent turns of 0.6K / 1.2K new tokens went 1.6 / 2.2 s -> 1.3 / 2.0 s with the setting, one run each), Windows, tensor-parallel groups.
