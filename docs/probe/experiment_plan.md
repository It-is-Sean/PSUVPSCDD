# Experiment plan from the current state

## Current plan override — 2026-05-03

Older phase labels below remain useful history, but the next valid plan is now:

1. **Use full SCRREAM, not `eval_scrream`.** The old local SCRREAM subset remains invalid for claims. The corrected branch uses `~/datasets/SCRREAM`.
2. **Use mesh-complete SCRREAM targets first.** The current default GT source is registered scene meshes sampled proportional to surface area, cropped to the selected two-view union frustum, and stored in the first input camera frame.
3. **Launch through Slurm.** Data generation and training scripts live in `slurm/`; logs go to `slurm_out/`.
4. **Generate before training.** The full adapter `.pt` generation job `85773` was running at 2026-05-03 02:26 CST, with dependent MLP training job `85774` pending. Verify shape / metadata / previews before interpreting training.
5. **Train the MLP baseline before method sprawl.** First baseline is `adapter_type=mlp`, `adapter_layers=4`, `adapter_hidden_dim=1024`, `loss_type=nova_flow`, `num_queries=10000`.
6. **Keep ScanNet as a diagnostic baseline.** All new ScanNet K-view trials must set `scannet_max_interval=1` unless the experiment explicitly studies wider baselines. Compare ScanNet checkpoints with fixed robust metrics before claims.
7. **Defer InteriorGS.** InteriorGS remains a plausible data-quality migration path, but it is not the immediate next branch.

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

## Phase 2 — Short ScanNet probe baseline

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

## Phase 5 — InteriorGS high-quality data pilot

InteriorGS-style high-quality indoor 3DGS data is deferred until after the corrected SCRREAM full-data baseline is generated, trained, and inspected.

Immediate steps:

1. download or stage a small InteriorGS subset on the server;
2. inspect scene assets (`3dgs_compressed.ply`, `labels.json`, occupancy files,
   and `structure.json`);
3. document coordinate transforms and metric units;
4. export a small target-point/render sanity set in the current probe convention;
5. create a fixed tiny train/val/test split and only then launch a training smoke.

See `interiorgs_training_plan.md`.

## Phase 6 — Corrected SCRREAM full-data branch

Current status:

1. full SCRREAM data is available locally at `~/datasets/SCRREAM`
2. data bridge exists at `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`
3. loader honors `--dataset scrream_adapter --data_root <adapter.pt>`
4. mesh-complete preview for `scene09` has been visually accepted
5. Slurm scripts exist for data preparation and MLP adapter training
6. checkpoints are staged under `checkpoints/` for NOVA `scene_n1`, `scene_n2`, `scene_ae`, and VGGT
7. SwanLab is installed in `nova3r`, and the full MLP Slurm script enables it by default
8. job `85773` is preparing the full adapter `.pt`; job `85774` is pending on `85773`

GT construction:

1. read official two-view pairs from `data/scrream/scrream_n2_list.json`
2. use the two RGB frames as adapter inputs
3. sample `sceneXX/meshes/*.obj` surfaces proportional to surface area
4. voxel-deduplicate and cache a per-scene mesh reservoir
5. crop points to the union frustum of the two input views
6. transform the target to the first input camera coordinate frame
7. FPS sample or pad to `10000` target points

Current Slurm chain:

```bash
squeue -j 85773,85774 -o '%.18i %.30j %.8T %.10M %.9l %.30R'
sacct -j 85773,85774 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList%20
```

Do not resubmit the default full prep/train jobs while `85773` / `85774` are still active.

Smoke variants can override script defaults through environment variables:

```bash
SCRREAM_ADAPTER_OUT=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_EXTRA_ARGS="--max_samples 8 --save_preview_dir experiments/probe3d/adapter_data/scrream_mesh_complete_smoke8_preview" \
sbatch slurm/scrream_mesh_complete_prepare.sbatch

SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_MAX_STEPS=100 \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/scrream_mesh_complete_smoke8_mlp \
sbatch slurm/scrream_mesh_complete_mlp_train.sbatch
```

## Phase 7 — Proposal-facing interpretation

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

1. Inspect `experiments/probe3d/probe_trials/results.tsv` rows beginning with `p4_`.
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

- `experiments/probe3d/result/probe_trials/p4_overnight_k2_mlp_l4_long_driver.out`
