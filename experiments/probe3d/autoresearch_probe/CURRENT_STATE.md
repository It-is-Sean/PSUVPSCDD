# Current PSUVPSC3DD probe state — 2026-04-29

This file is the compact current-state pointer for autonomous / autoresearch work.
It supersedes older numeric-only notes when conflicts appear.

## Non-negotiable corrections

1. **ScanNet interval correction**
   - Processed ScanNet root uses `frame_skip=20`.
   - Therefore `scannet_max_interval=1` means adjacent processed frames, i.e. roughly 20 raw-frame spacing.
   - Old runs using `max_interval=30` are interval-confounded and cannot support K-view conclusions.

2. **Metric correction**
   - Single scalar symmetric Chamfer and two-sample oracle averages are not reliable enough for claims.
   - Use fixed-sample robust metrics: pred→GT precision side, GT→pred recall side, F-score thresholds, trimmed CD, and visual videos.

3. **MLP baseline status**
   - The K2 interval=1 MLP-L4/chamfer checkpoint has some recall but poor precision/outlier behavior.
   - Fixed-5 robust eval for `p5_k2_adjacent_anchor_mlp_l4_chamfer_step1000/best.pth`:
     - symmetric CD mean/median: `0.0411 / 0.0326`
     - pred→GT mean distance median: `0.130`
     - GT→pred mean distance median: `0.0628`
     - F@0.05 mean/median: `0.328 / 0.355`
     - precision@0.05 mean: `0.238`
     - recall@0.05 mean: `0.536`
   - Interpretation: not just missing coverage; predictions are loose/outlier-heavy and visually poor.

   - Fixed-30 robust eval for the same checkpoint completed on 10 scenes x 3 samples (K2, interval=1):
     - symmetric CD mean/median/p90: `0.0504 / 0.0381 / 0.0639`
     - trimmed CD95 mean/median: `0.0335 / 0.0265`
     - pred→GT mean distance mean/median: `0.151 / 0.142`
     - GT→pred mean distance mean/median: `0.069 / 0.067`
     - F@0.05 mean/median: `0.291 / 0.275`; precision@0.05 mean `0.204`; recall@0.05 mean `0.532`
     - worst sample is `scene0000_02_00154/00155` with F@0.05 `0.0266`, confirming severe precision/outlier failures on some rows despite decent recall on others.
     - Outputs: `experiments/probe3d/result/autoresearch_probe/p5_k2_adjacent_anchor_mlp_l4_chamfer_step1000/robust_eval_fixed30/summary.json`; representative best/median/worst renders in sibling `robust_eval_fixed30_representative_renders/`.
     - Failure-case audit now also available: `experiments/probe3d/result/autoresearch_probe/p5_k2_adjacent_anchor_mlp_l4_chamfer_step1000/robust_eval_fixed30/failure_case_audit/failure_case_audit.md`. It ranks fixed-30 rows and confirms the dominant symptom is prediction-side precision/outliers: median recall-minus-precision gap at tau=0.05 is `0.3126`, pred→GT p90 correlates strongly with F@0.05 (`r=-0.8263`), and the worst row is `scene0000_02_00154/00155` with F@0.05 `0.0266`.

4. **GT quality correction**
   - GT-only visual audit showed the three ScanNet target modes are not dramatically different by eye.
   - Current failure should not be blamed solely on dirty GT.

## Active local autopilot

AutoResearchClaw full-pipeline execution is retired for this project. Its previous run drifted into irrelevant CIFAR/KD/FitNet work, so it should only be treated as a rejected artifact/reference, not as an experiment driver. Current autonomous progress uses the local harness under `experiments/probe3d/autoresearch_probe/` and the OpenClaw cron `PSUVPSC3DD local autopilot` every 15 minutes.

Local autopilot guardrails:

- stay inside the proposal: frozen visual backbone/VGGT-style features, lightweight or structured adapter, frozen NOVA3R-style complete-3D decoder;
- use corrected ScanNet `scannet_max_interval=1`;
- make claims only from fixed-sample robust/visual-first evaluation, not old `max_interval=30` runs, two-sample oracle rankings, or single-sample Chamfer;
- keep outputs under `experiments/probe3d/result/` unless source/docs changes are intentional.

## Next clean research step

1. Use the fixed-30 robust eval + best/median/worst renders to diagnose what differentiates success/failure cases.
2. Compare at least one proposal-aligned candidate under the same robust protocol: either a structured/query-conditioned adapter checkpoint or a pseudo-GT/token-distillation diagnostic.
3. Keep source/docs tidy and commit small logical updates on `wip/psuvpsc3dd-autoresearch-20260429`.
