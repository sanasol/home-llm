# DS (Maya) prefill A/B kit — rig ~/ds-pf, built 12.11.2026

Engine with PR #959 (`MAYA_DS4_PREFILL_CPU`): `~/maya-pf` = git worktree of `~/maya` at v1.1.61 (c5842c5)
+ cherry-pick of mw00/project-maya#959 (5ea06e1f), branch `pf-959`; build `~/maya-pf/build-pf/maya-ds4`
(`~/maya-pf/build-pf.sh`). The live `~/maya/build/maya-ds4` is untouched.

Arms (`mk_arms.py` writes `~/ds-pf/<arm>.json` from the live `maya-ds4.json`; only exe + env differ):
a-base · b-copy (FLOW_COPY_DMA + AI_RUN, bit-exact) · c-yoco (+YOCO_FLOW) · d/e (+ split 14,9,17 / 20,8,12)
· f-pcpu (+PREFILL_CPU, PR build) · g (EXPERT_MS 1.2) · h (+RESTORE_POINTS=0).

Run (DS must be idle; stops maya-ds, restores it and its expert profile on exit):
  ~/ds-pf/run_ab.sh                          # all arms, 2 rounds, cold 24/128/400K (~1.5-2 h)
  COLD=24,128 ROUNDS=1 ~/ds-pf/run_ab.sh a-base c-yoco f-pcpu   # quick pass (~25 min)
  python3 ~/ds-pf/summary.py                 # medians per arm
