# 28 — Strata 0.2.11 on the rig (engine 0.1.57.1 = v0.2.11 + PR #3341, control 0.1.57 + PR #3565), 06.11.2026

## Result
All three lanes run the vendor 0.2.11 stack with our two un-merged PRs re-applied. Downtime 02:58–03:02 CET (4 min,
all lanes), then three single-lane restarts of main (batch-mtp / DFlash2 arm / revert) and ~13 min of main down for
`strata tune auto`.

| | before (engine 0.1.56 = v0.2.9+PR3341) | after (0.1.57.1) | after + `--batch-mtp` (LIVE) |
|---|---|---|---|
| main RU (8 prompts) | 198.4 t/s, accept 0.68 | 205.8 | 201.9 |
| main RU thinking | 194.5 | 201.9 | 194.8 |
| main EN | 186.6 | 194.4 | 194.5 |
| main code-edit | 215.3 | 224.2 | 222.7 |
| main 2 parallel (per stream / sum) | 87.4 / 165.9, drafts 0 | 87.8 / 166.1 | **99.5 / 184.4 (+11%)** |
| main cold prefill 10K / 39K / 78K | 5234 / 5938 / 6981 | 5108 / 5920 / 6989 | — |
| lane2 solo (2 prompts) | 113 / 121 | 109 / 108 | — |
| lane3 solo | 113 / 119 | 123 / 111 | — |

Solo decode on main +3–4% (IQ3_S decode-math change + FP16 weight cache, both bit-identical per the release notes);
prefill unchanged (we already ran `--prefill auto:32768`); 3090 lanes within noise (n=2). Benches: `strata_spec_bench.py`,
`strata_lane_bench.py --n 2`, `strata_prefill_probe.py 8 32 64`; raw output on the rig in `~/bench-0211/`.

Checks after the switch: tool call (`finish_reason=tool_calls`, valid args) on :8001/:8003/:8004; auto-sessions line
("auto sessions on: keep 12 files ...") in all three `runtime/launch-*.server.log`; PR #3341 disk conversation cache under
`--batch 2 --batch-mtp`: 4 × 16K cold conversations, then the first again → "took back 16389 tokens from disk" in 365 ms,
request wall 0.46 s vs 2.6–3.0 s cold (`~/bench-0211/disk_cache_check.py`).

## Findings
1. **`--batch-mtp` had been missing on main since 02.11.** The single-slot A/B removed it; the return to `parallel: 2`
   (AGENTS 2.4) did not put it back, so parallel requests ran without MTP (drafts 0). Restored: +11% aggregate.
2. **DFlash2 (now default drafter upstream) loses here.** Same bench, `drafting: dflash2` (launcher picked
   `--draft dflash2 --spec 6 --draft-conf 0.8`, 2 slot drafters): RU 177 (−12%), RU-think 190 (−3%), EN 198 (+2%),
   code 221 (≈), parallel 90 / 165 (−10%). Acceptance is higher (0.79 RU), but no cyrillic draft vocab → slower RU.
   Stay on MTP + cyrillic vocab + `--batch-mtp`.
3. **`strata tune auto` (quick, coding) found nothing new.** One candidate (kv_resident_64k) not tried: needs 23.1 GiB
   free VRAM, 23.0 free. It waited ~5 min for the 3090 lanes to be "quiet" before measuring.
4. **The all-cards split engine cannot run on this rig** (control `strata_control/split_engine.py` rule): cards must be
   the same model (5090 + 3090 is out), ALL routed experts must fit in VRAM (IQ3_S experts ~50 GB > 2 × 24 GB for a
   3090-only split), context ≤ 128K; the package ships 2-card binaries only (`strata-split-2-sm86`, `-sm120`).
   Revisit only with two identical big cards.
5. Not applicable / deferred: K-quant MMQ prompt kernels (UD-Q4/5/6_K_XL only; we run IQ3_S); Unsloth v3.1 GGUFs
   (stock quants, we run the abliterated SC117 IQ3_S); Q8_0 n-gram table (+25 GB RAM, MemAvailable is 4–9 GB);
   llama.cpp `--moe-cache-mib` (no shared arena → 3 × 50 GB experts).

## How it is built (repeat for the next vendor release)
1. Engine: `~/build/Strata-0211` = git worktree of tag `v0.2.11`, branch `f-rig-0211`, `git merge pr-3341` (clean).
   `~/build/build-0211.sh`: cmake `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120;120-virtual"
   -DSTRATA_BUILD_CONVERSATION_TESTS=ON`, target `strata` (+ conversation tests: 253 + 4859 checks pass). ~6 min with ccache.
2. With all lanes stopped: `strata update` (signed feed; installs vendor engine 0.1.57 + control 0.1.57, test start ~25 s).
   The updater refuses while any engine serves.
3. `rig/patches/mk_rig_engine_pkg.py 0.1.57 0.1.57.1 ~/build/Strata-0211/build/strata` — hardlinked copy of the vendor
   package with our binary, unsigned, version 0.1.57.1 so the resolver prefers it.
4. Control: `patch` `components/control/0.1.57/serve/server.py` with `rig/patches/pr3565-auto-sessions-on-0.1.57.diff`
   (rebased from the 0.1.55 diff; all hunks apply, the vendor file equals `serve/server.py` of the tag; original kept as
   `server.py.orig-pre-pr3565`).
5. Start main → lane2 → lane3; check `engine started: .../0.1.57.1-.../strata` in `launch/<lane>.engine.log`.

## Rollback
- Engine: move `components/engine/0.1.57.1-linux-x86_64-cuda-cuda13` out → vendor 0.1.57 (but it does not know
  `--conversation-cache-disk-mib`/`--conversation-cache-dir`: drop them from main's preset first). The previous PR build
  (0.1.56 = v0.2.9 + PR #3341) is kept as `~/pr3341/pkg-0.1.56-live-20261106_025814` and still sits in components
  (resolver ignores it as lower).
- Control: `components/control/active.json` back to 0.1.55 (its patched server.py is intact).
- Presets: `~/.config/strata/launch-presets/*.json.bak-pre0211-20261106_025814`, launch state `~/.local/state/strata/launch/*.json.bak-pre0211-*`.
