# Experiment plan from the current state

## Current plan override — 2026-04-29 afternoon

Older phase labels below remain useful history, but the next valid plan is now:

1. **Stabilize evaluation.** Use corrected `scannet_max_interval=1`, fixed scene/view/sample manifests, robust PLY metrics, and representative renders.
2. **Treat MLP as baseline/failure mode.** Do not keep sweeping MLP losses just because CD changes.
3. **Compare proposal-aligned alternatives.** Test structured cross-attention/query-conditioned adapters or pseudo-GT / token-distillation diagnostics under the same fixed robust protocol.
4. **Let AutoResearchClaw organize, not redefine, the project.** ResearchClaw may write plans/stage artifacts, but it must respect the proposal: frozen visual backbone, frozen NOVA3R-style decoder, lightweight adapters, complete/amodal sparse-input-frustum reconstruction, visual-first robust evaluation.
5. **Supervisor gate.** Every 15 minutes, OpenClaw audits ResearchClaw process/logs/artifacts/diffs and corrects drift early.


This plan is intentionally short and tied to what is already real in the repo.

## Phase 0 — Corrections already absorbed

### SCRREAM correction
- older local `eval_scrream` runs are invalid for formal claims
- keep them only as engineering/debug history
- do not use them as feasibility evidence

### ScanNet path correction
- do not describe the ScanNet v2 line as official NOVA3R recipe reproduction
- describe it as a **NOVA3R-style extension / transfer probe**

## Phase 1 — Data and training infrastructure

### Completed
- full ScanNet v2 mesh-first preprocess
- formal `train / val / test` split
- scene-level `500k` mesh reservoirs
- true 4-view input wiring
- `pts3d_complete` dataset path
- DDP / `torchrun` conversion for MLP / CA / SA
- complete-GT smoke validation
- full-root DDP preflight validation

## Phase 2 — Autoresearch-style ScanNet probe baseline

### Goal
Quickly identify whether the failure comes from target definition, adapter capacity, or objective mismatch.

### Current best baseline
- dataset root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`
- target mode:
  - `anchor_frustum`
- adapter:
  - `MLP-L4, hidden=1024`
- loss:
  - direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best schedule:
  - `lr=5e-5` to step2000
  - `lr=1e-5` refinement to step2500
- best validation CD:
  - `0.08745259`

### Interpretation
This is a real numeric improvement, but not a visually clean reconstruction. The model has acceptable GT coverage but poor prediction precision / outlier control.

## Phase 3 — Paper-aligned NOVA3R target / loss reset

Jiacheng clarified the key methodological point: NOVA3R itself trains on complete / amodal point clouds **inside the selected input-view frustum**, not the entire room. The current `anchor_frustum` result was useful because it partially matched that idea, but for K-view training the target should be the union of the selected input frusta, not only the first view.

Immediate implementation plan:

1. add explicit target aliases:
   - `nova_input_frustum`: mesh surface points inside the union of selected input-view frusta
   - `nova_anchor_frustum`: first-view-only debug / K=1 equivalent
2. use NOVA-native target sampling through `src_complete_fps_*`, starting with `src_complete_fps_4096`
3. run paper-aligned K=1 and K=2 oracle sanity checks:
   - `p2_oracle_nova_input_frustum_k1_fps4096_s2_step400`
   - `p2_oracle_nova_input_frustum_k2_fps4096_s2_step400`
4. only after oracle support is plausible, run native-flow adapter probes:
   - `p2_adapter_nova_input_frustum_k1_mlp_l4_flow_fps4096_step1000`
   - `p2_adapter_nova_input_frustum_k2_mlp_l4_flow_fps4096_step1000`
5. keep direct Chamfer as a diagnostic / metric-chasing upper check, not the main claim

## Phase 3b — Precision-aware metrics after native-flow baseline

After the paper-aligned native-flow baseline is established, add precision-aware metrics/losses only as diagnostics or small auxiliaries:

1. log one-way distances:
   - `pred→GT` for precision / outlier control
   - `GT→pred` for completeness / recall
2. use GT-vs-pred videos for qualitative checks
3. consider weighted / trimmed Chamfer only if it does not replace the NOVA-native flow objective as the primary training signal

## Phase 4 — Adapter comparison on the same objective

After the paper-aligned native-flow MLP baseline is meaningful:

1. launch CA on the same processed data / target / objective
2. launch SA on the same processed data / target / objective
3. compare them only after all branches have comparable runs and visualizations

## Phase 5 — Return to corrected SCRREAM

After the ScanNet branch is stabilized:

1. wait for the official full SCRREAM dataset root
2. reconnect the corrected SCRREAM line to the proper data
3. rerun the adapter branches under the correct dataset assumptions

## Phase 6 — Proposal-facing interpretation

Only after the above:

- decide whether MLP remains only a baseline or becomes a stronger proposal anchor
- decide whether CA / SA materially strengthen the proposal
- update the proposal narrative with only the valid, corrected runs

## Phase 3c — Overnight K=2 feasibility sweep

Current tactical priority: produce a feasibility result rather than block on exact hidden NOVA3R GT construction.

### Fixed settings for the main overnight branch

- input views: `K=2`
- adapter: `MLP-L4-H1024`
- queries: `2048`
- main stable target: `anchor_frustum`
- training length: `1000` steps per trial
- validation: steps `500` and `1000`

### Loss ablation now running

1. `p4_k2_anchor_mlp_l4_flow_step1000`
2. `p4_k2_anchor_mlp_l4_hybrid005_step1000`
3. `p4_k2_anchor_mlp_l4_chamfer_step1000`

The purpose is to compare:

- generator-native flow matching
- flow matching with small rollout Chamfer auxiliary
- direct Chamfer as a diagnostic / upper-bound metric chaser

### GT-construction candidates queued after the loss ablation

Oracle sweep:

- `nova_input_frustum`
- `covered_by_ge2`
- `anchor_frustum_margin1.5`
- `nova_per_view_frustum_anchor_zpos`
- `nova_per_view_ldi2`
- `nova_per_view_ldi4`
- `nova_per_view_ldi8`

Adapter sweep:

- `covered_by_ge2`: flow + hybrid005
- `nova_input_frustum`: flow + hybrid005
- `anchor_frustum_margin1.5`: flow + hybrid005

### Morning checklist

1. Inspect `experiments/probe3d/autoresearch_probe/results.tsv` rows beginning with `p4_`.
2. Check whether K=2 MLP-L4 improves over previous K=4 feasibility runs.
3. Compare oracle ceiling vs adapter result for each target.
4. Inspect generated PLYs for the best metric candidates.
5. Select one result as the feasibility proof and keep the rest as ablations / diagnostics.

### Step-count update

Per user request, overnight adapter trials were increased from `1000` to `2000` steps where possible. The first short native-flow K=2 / MLP-L4 run completed at step1000 with CD `0.91071419`; it is now only a short-run diagnostic. The active overnight suite uses:

- `p4_k2_anchor_mlp_l4_flow_resume_step2000`
- `p4_k2_anchor_mlp_l4_hybrid005_step2000`
- `p4_k2_anchor_mlp_l4_chamfer_step2000`
- `p4_k2_{covered_ge2,input_frustum,anchor_margin15}_mlp_l4_{flow,hybrid005}_step2000`

Driver log:

- `experiments/probe3d/result/autoresearch_probe/p4_overnight_k2_mlp_l4_long_driver.out`
