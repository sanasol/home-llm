# Snapshot: hyperqwen-b before the "clone of A" change (2026-10-28)

Backup of slot B's serving profile before rebuilding it as a copy of slot A
(user decision 2026-10-28: drop the KVarN 262K 2-3-parallel profile; B becomes
FP8 KV + MTP + 150K on its own 3090 #2, port 18021, LiteLLM name unchanged).

Rollback (pre-change profile):
1. `sudo cp /root/localllm/rig/snapshots/2026-10-28-hyperqwen-b/hyperqwen-b.service.pre-clone /etc/systemd/system/hyperqwen-b.service` (on the rig the snapshot copy lives at `/opt/rig/snapshots/2026-10-28-hyperqwen-b/`)
2. `sudo systemctl daemon-reload && sudo systemctl restart hyperqwen-b`
3. Cold start ~10-20 min (vLLM 0.31 precompiled kernels, 150K/262K KV pool build).

Contents:
- `hyperqwen-b.service.pre-clone` — systemd unit as found (KVarN 4/2-bit KV, 262K, MTP, port 18021, GPU #2).
- `hyperqwen-a.service.reference` — slot A's unit (the clone target).
- `hyperqwen-b.service.pre-0.32` — clone-of-A unit as deployed on vLLM 0.31 (the 0.31 fallback for B).
- `hyperqwen-b.service.032-mtp` — the LIVE unit since 2026-10-28: vLLM 0.32 worktree, FP8 KV, MTP, 150K, KV pool 204,336 tokens.
- `hyperqwen-b.service.032-dflash-ru` — experimental unit: 0.32 + RU DFlash2 drafter (lapalaba), KV pinned 4.1 GB → 96,680 tokens, MAX_LEN=96000. Rejected in the 2026-10-28 ladder (see bench/results/2026-10-28-dflash2-b032/COMPARISON.md).
- `hyperqwen-b.service.032-dflash-stock` — experimental unit: 0.32 + stock DFlash2 drafter, same KV/MAX_LEN. +44% vs MTP at 90K, neutral at 8K/40K; not adopted as the B default (adoption rule: win in every window).
- `argv-observed.txt` — live `/proc/<pid>/cmdline` of both vLLM processes (truncated at capture; defines the effective flags).
- `env-and-state.txt` — /opt/rig/side.env (key redacted), vLLM version, venv, git state, CTX→MAX_LEN mapping in start_qwen.sh.

Rollbacks (all on the rig; snapshots dir = /opt/rig/snapshots/2026-10-28-hyperqwen-b/):
- B → MTP 0.32 (the default since 2026-10-28): `sudo cp .../hyperqwen-b.service.032-mtp /etc/systemd/system/hyperqwen-b.service && sudo systemctl daemon-reload && sudo systemctl restart hyperqwen-b`
- B → 0.31 clone-of-A: same with `hyperqwen-b.service.pre-0.32`.
- B → original KVarN 262K: same with `hyperqwen-b.service.pre-clone` (cold start ~10-20 min).
