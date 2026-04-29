# PSUVPSC3DD local autopilot prompt

Run every 15 minutes. You are 177 advancing the PSUVPSC3DD project directly, not through AutoResearchClaw.

## Source of truth

- `/home/jcd/PSUVPSC3DD_repo/PROPOSAL.md`
- `experiments/probe3d/autoresearch_probe/CURRENT_STATE.md`
- `experiments/probe3d/autoresearch_probe/heartbeat_log.md`
- `experiments/probe3d/autoresearch_probe/GIT_ORGANIZATION.md`

## Hard direction constraints

Stay within the proposal:

- frozen VGGT/VGGT-style visual backbone;
- lightweight/structured adapter;
- frozen NOVA3R-style complete-3D decoder;
- complete/amodal sparse-input-frustum 3D reconstruction;
- corrected ScanNet interval: `scannet_max_interval=1` for `frame_skip=20` data;
- visual-first fixed-sample robust evaluation;
- no claim-level reliance on old `max_interval=30`, two-sample oracle rankings, or single-sample Chamfer.

Do **not** run AutoResearchClaw full pipeline. Its previous full run drifted into CIFAR/KD/FitNet and is rejected. Use our local `experiments/probe3d/autoresearch_probe/` harness instead.

## Operating rules

1. Check for active GPU jobs. Avoid duplicate overlap, but GPU experiments around 30 minutes are allowed when useful.
2. Make one concrete project step:
   - run/monitor fixed-sample robust evaluation;
   - render representative samples;
   - prepare or run a small structured-adapter / pseudo-GT control;
   - tidy documentation if the research state changed;
   - or debug a blocker.
3. Keep outputs under `experiments/probe3d/result/` or committed project files under the relevant source/docs paths.
4. If tracked code/docs are modified, run relevant checks and commit a small logical commit on branch `wip/psuvpsc3dd-autoresearch-20260429` when safe. Do not commit secrets or result artifacts.
5. Append a concise entry to `experiments/probe3d/autoresearch_probe/heartbeat_log.md` with Checked / Action / Result / Next.
6. Return a concise Feishu update every run.

## Current priority queue

1. Expand robust eval from 5 samples to the fixed 30-sample manifest for the current MLP baseline.
2. Render representative success/failure cases from that fixed eval.
3. Compare a proposal-aligned candidate (cross-attention/query-conditioned adapter or pseudo-GT/token-distillation diagnostic) under the same protocol.
4. Keep repo clean and commit meaningful code/doc changes so Jiacheng can pull from another server.
