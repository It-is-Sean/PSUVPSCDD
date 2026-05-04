# Current PSUVPSC3DD probe state — 2026-04-29

This file is the compact current-state pointer for PSUVPSC3DD probe work.
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
     - Outputs: `experiments/probe3d/result/probe_trials/p5_k2_adjacent_anchor_mlp_l4_chamfer_step1000/robust_eval_fixed30/summary.json`; representative best/median/worst renders in sibling `robust_eval_fixed30_representative_renders/`.
     - Failure-case audit now also available: `experiments/probe3d/result/probe_trials/p5_k2_adjacent_anchor_mlp_l4_chamfer_step1000/robust_eval_fixed30/failure_case_audit/failure_case_audit.md`. It ranks fixed-30 rows and confirms the dominant symptom is prediction-side precision/outliers: median recall-minus-precision gap at tau=0.05 is `0.3126`, pred→GT p90 correlates strongly with F@0.05 (`r=-0.8263`), and the worst row is `scene0000_02_00154/00155` with F@0.05 `0.0266`.


4. **Cross-attention candidate status**
   - `p7_k2_i1_anchor_ca_l2_h512_chamfer_step1000` completed 1000 steps with K2, `scannet_max_interval=1`, `anchor_frustum`, cross-attention L2/H512, and `chamfer_sample`.
   - Validation CD at step 1000 was `0.54222615`, so this is not a scalar improvement over the MLP-L4/chamfer baseline.
   - A fixed-30 robust eval was launched at `experiments/probe3d/result/probe_trials/p7_k2_i1_anchor_ca_l2_h512_chamfer_step1000/robust_eval_fixed30/`; use that plus renders before interpreting whether cross-attention improves precision/outlier behavior.

5. **GT quality correction**
   - GT-only visual audit showed the three ScanNet target modes are not dramatically different by eye.
   - Current failure should not be blamed solely on dirty GT.

## Active direction

The ScanNet probe has isolated the current failure mode: the best corrected
MLP baseline has moderate recall but poor prediction-side precision and
outlier control. The next planned training direction is to test whether
InteriorGS-style high-quality indoor 3DGS supervision gives a cleaner data
signal on this server.

InteriorGS migration guardrails:

- do not treat InteriorGS as a drop-in ScanNet replacement; it is 3DGS data and needs a data bridge;
- first inspect `3dgs_compressed.ply`, `labels.json`, occupancy files, and `structure.json` on a small subset;
- document coordinate frames, metric units, and any transform into the current probe convention;
- create fixed tiny splits and visual sanity sheets before training;
- keep ScanNet fixed-30 metrics as diagnostic baselines, not the main next data direction.

## Next clean research step

1. Stage/download a small InteriorGS subset on this server.
2. Write a minimal inspection script for scene assets and coordinate bounds.
3. Export one visual sanity artifact per pilot scene.
4. Only after that, launch an adapter-training smoke on the new data bridge.
5. Keep source/docs tidy and commit small logical updates on `wip/psuvpsc3dd-probe-20260429`.
