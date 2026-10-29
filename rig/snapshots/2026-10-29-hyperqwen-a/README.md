# hyperqwen-a snapshots — 2026-10-29 A -> vLLM 0.32 rollout

- hyperqwen-a.service.pre-0.32 — A before the 0.32 rollout (vLLM 0.31 main tree ~/HyperQwen, FP8 KV, MTP, 150K, :18020, GPU-9c7268f4).
- hyperqwen-a.service.032 — live unit: same profile, 0.32 worktree ~/HyperQwen-0.32 (shared venv with B).

Rollback: sudo cp hyperqwen-a.service.pre-0.32 /etc/systemd/system/hyperqwen-a.service && sudo -S -p '' systemctl daemon-reload && sudo -S -p '' systemctl restart hyperqwen-a
Also kept on rig: /etc/systemd/system/hyperqwen-a.service.bak-0.31-20261029.
Verified 29.10: health 200 in ~30s, KV 204,336, MTP accept ~45%, deterministic probe identical to B-0.32, real_rep e2e avg ~99.7 tok/s (A-0.31 was 96.3, B-0.32 102.2).
