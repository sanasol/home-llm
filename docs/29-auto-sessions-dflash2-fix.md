# 29 — auto-sessions broken by DFlash2 drafting (0.2.11), fixed 06.11.2026

## Symptom
Since the 0.2.11 rollout (06.11 03:00, docs/28), every auto-sessions park on the 3090 lanes
failed: `auto sessions: conversation snapshot: invalid K/V extent` (53x on lane3 in 15 min).
No `auto-<conv>.bin` was ever written, so with two conversations sharing a 1-slot lane every
switch paid a FULL re-prefill (21-22 s at 64K tokens) instead of a ~0.5 s disk restore.
Compounded by the pool-balancing bug fixed the same day (rig_affinity load-based placement,
commit 123ed07) — two agents pinned to one lane + broken park = constant thrash.

## Root cause
Strata 0.2.11 introduced the `drafting` run-config choice; `auto` (the new default) resolves to
**DFlash2** for Qwen3.8-Flash-Next (`strata_control/drafting.py`). The 03:02 lane launches rewrote
the presets: lane2/lane3 started `--draft dflash2 --draft-fallback none` and NO `--mtp` (docs/28
had rejected DFlash2 for RU — that decision was encoded only in main's args, and `drafting: null`
let the launcher re-decide). The PR #3565 session-file path in `serve` passes the drafter's KV
state unconditionally (`mtp.kv_state()`); with MTP not loaded that is an empty `QsaState`
(max_cells=0, kv_mode=0), and `conversation_snapshot.cpp:valid_extent` rejects
`upto(=conversation tokens) > max_cells(=0)` — every SAVE died. Every other engine call site
already guards with `use_mtp ? &mtp.kv_state() : nullptr`; the session-file path was the exception
(latent upstream bug, exposed by our drafting flip).

## Fix (engine 0.1.57.2 = f-rig-0211 + `snapshot-dflash2-draft-guard-0.2.11.diff`)
1. `generate.cpp`: the five session-file call sites (SAVE sources, RESTORE read-limits/validate/
   restore, disk-cache read-limits) now pass `o.mtp.empty() ? nullptr : &mtp.kv_state()`.
2. `conversation_state.cpp` + header: pointer overload of `conversation_session_read_limits`
   (layers-only limits when `draft == nullptr`).
3. The two extent error strings now print `upto/max_cells/n_pages/kv_mode` (diagnostics kept).
4. Presets: `drafting: "mtp"` pinned in lane2/lane3/main launch presets (repo copies synced,
   api_key masked) — restores the cyrillic-draft-vocab MTP drafter (docs/28: DFlash2 is -12% RU).

## Verified (lane2, same-day)
- dflash2 + guard: saved 64244 tokens in 371 ms, restored in 501 ms, switch-back turn **1.2 s**
  vs 21.5 s re-prefill (probe: two 64K-token conversations alternating, `/tmp/snap_probe.py`).
- mtp + guard: saved/restored with the draft part (1.217 GB files), restore 516 ms, turn 1.2 s.
- Old pre-abliterated session files are rejected cleanly ("saved with another model"); prune keeps
  12 files / 100 GB.

## Notes
- main still runs 0.1.57.1 (MTP loaded, batch mode never reaches the session-file path) — it picks
  up 0.1.57.2 on its next restart; the resolver already prefers it.
- Session files saved by a dflash2 engine (layers-only) do not restore into an MTP engine
  (K/V layer-count mismatch) and vice versa: one cold prefill after a drafting flip, by design.
- Next vendor update: rebase BOTH `pr3565-auto-sessions-on-*.diff` (control) and
  `snapshot-dflash2-draft-guard-0.2.11.diff` (engine, branch f-rig-0211 worktree ~/build/Strata-0211)
  before restarting lanes; after any drafting/auto change, grep `launch-<lane>.server.log` for
  `invalid K/V extent` and confirm `session saved ... in ... ms` lines before trusting a lane.
- Rollback of the engine part: move `components/engine/0.1.57.2-*` out (back to 0.1.57.1); the
  presets stay on mtp, which alone already fixes the symptom on 0.1.57.1.
