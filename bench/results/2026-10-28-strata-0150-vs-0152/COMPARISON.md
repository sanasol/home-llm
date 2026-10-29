# Strata engine 0.1.50 → 0.1.52 (release v0.2.7) — before/after, 2026-10-28

Rig: hub 5090 (32 GB), model Qwen3.8-Flash-Next IQ3_S (GSQ-RCO, 2-shard GGUF), service `llm-main` :8001.

## Runs

| | baseline | after |
|---|---|---|
| run id | 20261028T220935Z-129b7382 | 20261028T231624Z-36cf1a50 (see results.json) |
| engine version | 0.1.50 (build 1aff6d14117b+cuda13.0) | 0.1.52 (build a42cc3d7df70+cuda13.0) |
| strata commit | 562a0c2ed7169ff6373f893b162106d55a828297 | 55a557bb6d77facc73d291c3f9894289496c770c |
| protocol | strata-qwen38-thinking-v1-draft.1 (draft), standard suite | same |
| status | complete, conforming | complete, conforming |
| quality | 12/12 (4 instruction, 4 retrieval, 4 tool_call) | 12/12 |
| MTP acceptance (all 22 requests) | 2695/4009 = 67.2% | 2755/4166 = 66.1% |
| environment | Ubuntu 26.10, driver 615.71.09, CUDA 13.4, RTX 5090 32GB @460W, 9950X 96GB | same |

Engine flags (identical in both runs, from `llm-main` preset `main`): `--pack iq3_s --native 00001-of-00002.gguf --ple-gguf 00002-of-00002.gguf --expert-profile terminal/0.1.51/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp /models/mtp/rt --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700 --adapt-async 1 --pcie-frac 0.35 --batch 2`.

Both engines were freshly loaded before their run (service stop → start). The update itself: `strata update --json` (update-0.1.50-to-0.1.52.json), signed download, full test-start of 0.1.52 answered a probe, then activated. 0.1.50 folder kept for rollback.

## Performance (medians of 3 repeats; min/max in parentheses)

| stage | metric | 0.1.50 | 0.1.52 | Δ median |
|---|---|---|---|---|
| short | prompt_tps | 191.26 (178.9–236.0) | 185.50 (179.2–245.1) | −3.0% |
| short | decode_tps | 183.95 (173.7–184.5) | 186.82 (184.7–187.2) | +1.6% |
| short | ttft_s | 0.412 (0.338–0.443) | 0.425 (0.323–0.442) | +3.1% |
| short | first_answer_s | 0.918 (0.654–1.363) | 0.829 (0.780–0.907) | −9.7% |
| short | total_s | 1.953 (1.868–2.873) | 2.179 (1.911–2.533) | +11.6% |
| 4K | prompt_tps | 3064.5 (3054.0–3198.9) | 3106.4 (3097.9–3143.9) | +1.4% |
| 4K | decode_tps | 159.19 (159.0–181.7) | 172.94 (162.0–190.0) | +8.6% |
| 4K | ttft_s | 1.347 (1.291–1.355) | 1.329 (1.312–1.334) | −1.3% |
| 4K | first_answer_s | 3.700 (3.181–4.219, n=2) | 2.754 (n=1) | −25.6% |
| 4K | total_s | 4.499 (4.162–4.559) | 4.266 (4.019–4.481) | −5.2% |
| 16K | prompt_tps | 5656.0 (5651.5–5700.2) | 5714.3 (5694.3–5726.7) | +1.0% |
| 16K | decode_tps | 178.52 (162.9–181.7) | 174.02 (172.4–182.1) | −2.5% |
| 16K | ttft_s | 2.915 (2.892–2.915) | 2.883 (2.876–2.897) | −1.1% |
| 16K | first_answer_s | 5.184 (4.881–5.487, n=2) | 5.223 (4.876–5.570) | +0.7% |
| 16K | total_s | 5.750 (5.725–6.048) | 5.811 (5.686–5.857) | +1.1% |

## Verdict

- No regression: quality 12/12 both, MTP acceptance ~67%→66%, TTFT flat, prefill +1% (noise).
- No meaningful win either: decode changes (+1.6% short, +8.6% 4K, −2.5% 16K) are within per-run variance (overlapping min/max bands); total_s differences are driven by reasoning length, not engine speed.
- The 0.2.7 release notes (routing experts → VRAM +15–37%) target ≤13 GB GPUs and dual-GPU prefill — neither applies to this 32 GB single-GPU 5090 config, which matches the near-zero measured delta.
- Keep 0.1.52 (signed, tested, conforming on both sides). Rollback path: `strata components rollback engine` (0.1.50 folder retained) if anything degrades over the following days.

`strata bench compare` (CLI) rejects these run dirs with "unsupported benchmark report" (expects a different report format); comparison above computed from results.json.
