# DFlash2 drafters on B (vLLM 0.32) vs the MTP baseline — ctx ladder, 2026-10-28

## Question
The user's Phase-2 ask: put the RU-tuned DFlash2 drafter (`lapalaba/Qwen3.8-27B-DFlash2-ru`,
3.6 GB) on slot B and check whether it beats B's current MTP speculative decoding.
Adoption rule (agreed 2026-10-27): **adopt only if it wins MTP in every context window** —
the stock drafter was rejected on 27.10 for exactly that reason (+11% @90K, −13% @40K,
+5% @8K on slot A).

Because the RU drafter is language-tuned, a control arm with the **stock** DFlash2 drafter
was run on the same engine the same day: it separates "0.32 dflash integration broken"
from "RU specialization bad for this workload".

## Protocol (`bench/ru_ctx_ladder.py` + `.sh`, added to the 0.32 worktree)
- Slot B, one 3090, three arms, each a full service restart of `hyperqwen-b`:
  - `b032-mtp` — the live MTP arm (unit `hyperqwen-b.service.032-mtp`, K=3 MTP tokens, KV 204,336, 150K)
  - `b032-dflash-ru` — `SPEC=dflash2 DRAFT=models/Qwen3.8-27B-DFlash2-ru` (unit `.032-dflash-ru`, K=7, KV pinned 4.1 GB → 96,680, MAX_LEN=96000)
  - `b032-dflash-stock` — same but `DRAFT=models/Qwen3.8-27B-DFlash2` (unit `.032-dflash-stock`)
- Workload: first 4 prompts of `bench/prompts_real.jsonl` (the production C1 coding mix),
  each padded **at the tail** to the target context (8K / 40K / 90K tokens) with a
  run-unique, non-repeating filler pool (140K tokens; per-(ctx,prompt,rep) windows →
  prefix cache provably defeated; measured input lens: 7,949–7,958 / 39,780–39,793 / 89,512–89,540).
- `vllm bench serve --custom-output-len 300 --max-concurrency 1 --temperature 0` per batch;
  decode tok/s from Mean TPOT (prefill excluded); tok/step = 1 + K·accepted/draft-tokens
  from the bench summary (0.32 prints Draft/Accepted inline); window tok/s = output over
  the whole batch (prefill included, so it is NOT comparable across ctx).

## Results (single rep per arm per window; raw lines in the three `.out` files)

| ctx | MTP (K=3) | RU dflash (K=7) | stock dflash (K=7) | RU vs MTP | stock vs MTP |
|---|---|---|---|---|---|
| 8K  | **103.5** t/s, 2.58 tok/step | 49.5, 2.76 | **104.4**, 3.75 | **−52%** | +0.9% |
| 40K | **99.2**, 2.68 | 21.6, 2.97 | **98.1**, 3.65 | **−78%** | −1.1% |
| 90K | 53.0, 2.82 | 64.9, 2.72 | **76.6**, 3.24 | +22% | **+44.5%** |

Per-position acceptance seen in the engine logs (tail of each window):
- MTP warmup: 0.64/0.49/0.41 (avg 51%); dflash-ru warmup (RU warmup prompts): 0.78/0.73/0.69/0.67/0.64/0.60/0.58 (avg 67%);
- dflash-ru @40K batch: 0.57/0.27/0.11/… (avg 16%) — collapses on the coding prompts;
- dflash-stock @90K batch: 0.94/0.84/0.81/0.77/0.77/0.74/0.68 (avg 79%) — excellent at long ctx.

## Verdict
- **RU drafter: rejected.** It is the wrong specialization for B's actual load: B is the
  par27 subagent lane, which runs mostly coding/English tasks. Its RU-tuned head collapses
  to 16% average acceptance on the coding mix (vs 67% on RU warmup text) and loses to MTP
  by 2–4.6× at 8K/40K; the +22% at 90K does not compensate (and the adoption rule demands
  a win in every window). The 3.6 GB of VRAM for it would shrink B's KV pool to 96,680
  tokens (vs 204,336) — pure loss on this workload.
- **Stock drafter on 0.32: the 0.32 dflash path is healthy** (same wall speed as MTP at
  8K/40K, +44.5% at 90K, no crash, KV 96,680). It does **not** meet the strict
  "win in every window" rule (−1.1% at 40K, within noise; +0.9% at 8K, within noise), so
  B stays on the MTP arm as the default. The 90K +44.5% is the interesting number: it is
  the regime where the target's own decode gets slow (90K attention) and speculation
  hides the draft cost. Keep the stock unit (`hyperqwen-b.service.032-dflash-stock`) in
  the snapshot dir for a future manual switch if long-input work becomes routine on B.
  Note the KV pool halves (204,336 → 96,680) with any dflash arm, so it is a
  concurrency-for-latency trade, not free.

## 0.32 integration notes (new findings)
- 0.32 prints `WARNING ... no KV cache group could be identified as the draft model's`
  on **both** dflash arms (RU and stock) — 0.31 did not. Stock still performs fine, but
  worth an upstream question to syv-ai/HyperQwen (PR #387) whether the drafter KV should
  be a separate group (and fp8'd by the FA2 patch) — could be why dflash drafts show
  `Add 3 padding layers, may waste at most 60% KV cache memory`.
- **Metric names changed in 0.32**: MTP counters are now
  `vllm:spec_decode_num_drafts_total` (steps), `..._num_draft_tokens_total`,
  `..._num_accepted_tokens_total` (+ per-position breakdown) — 0.31 used
  `..._num_draft_total` / `..._num_accepted_total`. Anything parsing B's /metrics for
  spec-decode stats (rig-monitor, dashboards) needs the 0.32 names for B while A still
  emits the 0.31 names.
- B-0.32 stop position on a deterministic 1024-token chat probe differed from A-0.31
  (700 vs 405 tokens, identical reasoning content) — different MTP/sampling internals
  between 0.31 and 0.32; not a correctness issue, noted for completeness.

## Rollback
B was left on the MTP arm (`hyperqwen-b.service.032-mtp`, restored and verified:
service active, KV 204,336, MTP counters live, LiteLLM `qwen3.8-27b-b` OK). All three
units are in `/opt/rig/snapshots/2026-10-28-hyperqwen-b/` and mirrored in this repo.
