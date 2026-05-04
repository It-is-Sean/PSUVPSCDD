# PSUVPSC3DD / Probe Workspace

## Current canonical status — 2026-05-03

This branch is now a **server-side research workspace**. The source of truth is:

- `PROPOSAL.md`
- `PROJECT.md`
- `experiments/probe3d/README.md`
- `docs/probe/handoff_2026-05-03.md`
- `docs/probe/experiment_plan.md`

Important corrections that override older sections below:

1. **ScanNet view interval:** processed data already uses `frame_skip=20`; corrected experiments use `scannet_max_interval=1` (adjacent processed frames, roughly 20 raw frames). Older `max_interval=30` runs are interval-confounded.
2. **Metric reliability:** single symmetric CD and two-sample oracle averages are diagnostic only. Current comparisons use fixed samples, pred-to-GT precision, GT-to-pred recall, F-score thresholds, trimmed CD, and representative renders.
3. **Current MLP baseline:** K2/interval=1 `anchor_frustum + MLP-L4 + chamfer_sample` is recall-heavy but precision/outlier-poor on fixed-30 robust eval: F@0.05 mean/median `0.291/0.275`, precision@0.05 mean `0.204`, recall@0.05 mean `0.532`.
4. **Latest structured-adapter check:** K2/interval=1 `anchor_frustum + cross_attention L2/H512 + chamfer_sample` completed 1000 steps with validation CD `0.54222615`, which is not better than the MLP baseline by scalar validation CD. Its fixed-30 robust eval should be inspected before any claim if result artifacts are available.
5. **SCRREAM full-data status:** the old `eval_scrream` package is still invalid for claims, but the full SCRREAM tree is now available locally at `~/datasets/SCRREAM`.
6. **Current SCRREAM GT path:** the active data bridge uses registered scene meshes (`sceneXX/meshes/*.obj`) as the complete target source, samples mesh surfaces proportional to surface area, crops to the selected two-view union frustum, transforms targets into the first input camera frame, and exports fixed `10000 x 3` adapter targets.
7. **Slurm convention:** long data generation and training jobs should be launched through scripts in `slurm/`, with logs in `slurm_out/`.
8. **Current Slurm status:** on 2026-05-03 02:26 CST, full SCRREAM mesh-complete prep job `85773` was running and dependent MLP training job `85774` was pending; the full adapter `.pt` had not yet been written.
9. **Local weights:** NOVA3R `scene_n1`, `scene_n2`, `scene_ae`, and VGGT weights are staged under `checkpoints/`; Slurm scripts default network proxy variables to `http://127.0.0.1:7896`.
10. **Git handoff:** the local cleanup branch is `wip/psuvpsc3dd-probe-20260429`; push it before using it for a fresh remote checkout.

This repository is currently a **research execution workspace** around a simple question:

> can frozen visual backbones (currently VGGT-first) drive a NOVA3R-style decoder through a lightweight adapter, and learn useful complete 3D reconstruction behavior under reliable supervision?

It contains:

- the NOVA3R / DUST3R style reconstruction stack used here as the decoder/data backbone
- the probe / adapter experiments under `experiments/probe3d/`
- proposal-facing documentation under `docs/probe/`

## Current experimental status

There are now three lines, and they should be interpreted differently.

### 1. SCRREAM full-data line

A retrospective audit found that the earlier local `eval_scrream` package was only the released **evaluation subset** (~1.6 GB), not the official full training-scale dataset.

So:

- the older SCRREAM quantitative results are **invalid as formal scientific evidence**
- they are still useful as **pipeline/debugging history**
- they should not be used as proposal feasibility claims

The corrected full-data branch is now active because the full SCRREAM tree is present at:

- `~/datasets/SCRREAM`

The current target source is **mesh-complete**, not the earlier eval-subset pseudo-GT and not the depth-only bridge by default:

- input samples come from `data/scrream/scrream_n2_list.json`
- RGB inputs are the two frames in each official pair
- GT points are sampled from `sceneXX/meshes/*.obj`
- mesh sampling is surface-area proportional, so walls / room structure are not underweighted relative to small objects
- targets are cropped to the union frustum of the two input views
- final `target_points` are stored in the first input camera coordinate frame
- the output `.pt` is consumed through `--dataset scrream_adapter --data_root <adapter.pt>`
- for `nova_flow`, SCRREAM targets are normalized with the NOVA `scene_ae` checkpoint `norm_mode`; the local `scene_ae` config reports `median_3`

Current job state recorded on 2026-05-03 02:26 CST:

- `85773` / `scrream_mesh_prep`: `RUNNING` on `air-node-04`
- `85774` / `scrream_mesh_mlp`: `PENDING (Dependency)` after `85773`
- target full adapter data was not yet written:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt`
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.manifest.json`

### 2. ScanNet v2 line

The ScanNet v2 mesh-first extension remains the current diagnostic baseline.

Important interpretation note:

- this is a **NOVA3R-style extension / transfer probe** on ScanNet v2
- it is **not** a literal reproduction of the official NOVA3R scene-training recipe, which is described around `3D-FRONT + ScanNet++V2`

The training structure remains aligned with the proposal direction:

- frozen visual backbone
- lightweight adapter
- NOVA3R-style decoder / flow-matching training path
- reliable **complete-GT style** supervision

### 3. InteriorGS line

InteriorGS remains a plausible future data-quality migration path, but it is no longer the immediate next step. The current priority is to finish the SCRREAM full mesh-complete data generation and first MLP adapter baseline first.

## Active ScanNet v2 mesh-first plan

The current formal ScanNet training path uses:

- geometry source: `vh_clean.ply`
- frame sampling: `frame_skip=20`
- GT definition: `mesh surface ∩ sparse-input-view union frustum`
- no extra visible/occluded auxiliary labels for now
- scene-level reservoir: `500k` mesh-sampled points per scene
- true **4-view input**
- DDP / `torchrun` training launchers

Formal plan details live in:

- `docs/probe/scannet_mesh_first_plan.md`

## What is already implemented

### Data / supervision
- full ScanNet v2 preprocess completed to:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k`
- formal split created at:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`
- split scene counts:
  - `train=1362`
  - `val=151`
  - `test=100`

### Model / training plumbing
- ScanNet now truly feeds **4 input views**
- `pts3d_complete` is available from scene-level mesh reservoirs
- MLP / CA / SA training scripts were converted to **DDP / torchrun**
- launchers were updated accordingly

### Validation status
- visible-depth ScanNet smoke passed earlier
- complete-GT ScanNet smoke passed
- full-root `torchrun --nproc_per_node=1` MLP DDP preflight also passed

## Current ScanNet probe result

The long formal 50-epoch MLP run is no longer the most informative immediate path. A short probe harness under `experiments/probe3d/probe_trials/` isolated the main failure mode more quickly.

Current best numeric run:

- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- objective: direct sampled rollout Chamfer, `loss_type=chamfer_sample`
- schedule: `lr=5e-5` to step 2000, then `lr=1e-5` refinement to step 2500
- best validation CD: `0.08745259`
- output dir: `experiments/probe3d/result/probe_trials/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

Important interpretation:

- switching from `nova_flow` to direct rollout Chamfer is what moved CD down from roughly `0.30–0.35` to `<0.1`
- this mostly fixes the training/evaluation objective mismatch
- the visual result is still not good enough: the prediction covers the GT but contains many loose / thick-shell / outlier points
- the problem has shifted from **recall / target reachability** to **precision / sharpness**


### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json`

## Research direction right now

The practical near-term plan is:

1. keep the old `eval_scrream` correction in mind and do not reuse those invalid claims
2. monitor Slurm prep job `85773` until the full SCRREAM mesh-complete adapter `.pt` and manifest exist
3. let dependent Slurm job `85774` start the first SCRREAM MLP adapter baseline only if prep succeeds
4. inspect losses, validation samples, and exported PLYs before making any adapter claim
5. keep the fixed-30 ScanNet metrics as a failure-mode baseline
6. defer InteriorGS until the corrected SCRREAM full-data baseline is understood

## Documentation map

- `AGENTS.md` — project-level instructions for future coding agents
- `PROJECT.md` — current project-level status and next steps
- `docs/probe/README.md` — probe-doc entry point
- `docs/probe/handoff_2026-05-03.md` — current machine handoff for SCRREAM mesh-complete
- `docs/probe/scannet_mesh_first_plan.md` — current ScanNet formal plan
- `docs/probe/experiment_history.md` — what really happened, including corrections
- `docs/probe/experiment_plan.md` — phased execution plan from here
- `docs/probe/interiorgs_training_plan.md` — deferred high-quality dataset migration plan
- `docs/probe/todo.md` — current actionable task list
- `slurm/` — Slurm job scripts; logs should go to `slurm_out/`
