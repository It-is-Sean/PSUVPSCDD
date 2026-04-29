# PROJECT.md

## Current canonical state — 2026-04-29 afternoon

AutoResearchClaw is now installed and running as an orchestration layer, with a 15-minute OpenClaw supervisor that audits process status, proposal alignment, and code diffs. The ResearchClaw integration lives under `researchclaw/`; generated ResearchClaw outputs stay under `experiments/probe3d/result/researchclaw/` and are gitignored by the existing result rules.

The current research interpretation is:

- old ScanNet experiments with `max_interval=30` are confounded because the processed data already used `frame_skip=20`; use `scannet_max_interval=1` for the intended NOVA3R-style spacing;
- the raw ScanNet GT audit did not show obviously catastrophic data quality;
- oracle/CD rankings were unstable and should not drive claims;
- the interval-corrected MLP baseline has coverage but poor precision/outlier control, so the next valid progress must improve robust visual-first metrics, not just symmetric CD.

The immediate project contract is described in `experiments/probe3d/autoresearch_probe/CURRENT_STATE.md`.


## Project identity

This repo is currently a **research execution workspace** for probing whether strong frozen visual features can be adapted into a NOVA3R-style scene reconstruction decoder with minimal trainable structure.

The present focus is intentionally narrow:

- image / geometry first
- reliable supervision first
- scale validation before method sprawl

## Current status

There are now two branches, with very different evidential status.

### A. SCRREAM branch

The earlier local `eval_scrream` experiments were later found to use only the released **evaluation subset**, not the official full SCRREAM training-scale dataset.

Therefore:

- those old SCRREAM numbers are **invalid as formal evidence**
- they remain useful as engineering/debug history only
- the correct SCRREAM rerun is still pending full-data availability

### B. ScanNet v2 branch

The current active formal line is a **ScanNet v2 mesh-first extension**.

It should be interpreted as:

- a **NOVA3R-style extension / transfer probe** on ScanNet v2
- not a literal reproduction of official NOVA3R training on `3D-FRONT + ScanNet++V2`

Still, this line is now serious enough to serve as the proposal’s **reliable-target baseline**.

## Formal ScanNet v2 setup

### Data / supervision
- mesh source: `vh_clean.ply`
- frame extraction: `frame_skip=20`
- GT definition: `mesh surface ∩ sparse-input-view union frustum`
- no extra visible/occluded auxiliary labels in this version
- scene-level complete reservoir: `500k` points / scene

### Processed dataset
- processed root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k`
- split root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`
- scene counts:
  - train = `1362`
  - val = `151`
  - test = `100`

### Training stack
- true 4-view input
- `pts3d_complete` available in dataset batches
- MLP / CA / SA launchers converted to `torchrun` / DDP

## Current probe baseline

The most informative current run is the short autoresearch-style ScanNet probe, not the old long formal MLP schedule.

### Best numeric configuration
- target mode: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- decoder: frozen NOVA-style scene decoder
- training objective: direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- training schedule:
  - `lr=5e-5` to step 2000
  - resume with `lr=1e-5` to step 2500
- best validation CD: `0.08745259`
- output dir:
  - `experiments/probe3d/result/autoresearch_probe/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

### Interpretation
- `nova_flow` objective was mismatched with the final sampled-point Chamfer evaluation.
- Direct rollout Chamfer pushed validation CD below `0.1`.
- Visualized point clouds remain visibly noisy / thick / outlier-heavy.
- Current bottleneck is therefore not just CD minimization, but prediction precision and distribution sharpness.

## What has already been validated

- ScanNet visible-depth smoke passed
- ScanNet complete-GT smoke passed
- full-root `torchrun --nproc_per_node=1` MLP DDP preflight passed
- mesh / pose alignment looks valid on sampled scenes
- full preprocess completed successfully

## Immediate next step

1. keep `anchor_frustum + direct Chamfer` as the numeric baseline
2. implement a precision-aware loss variant, e.g. overweight `pred→GT` or use trimmed / outlier-aware Chamfer
3. evaluate with both CD and side-by-side GT/pred point-cloud videos
4. only after visual quality improves, compare CA / SA adapters on the same target/objective
5. later, rerun SCRREAM only after correct full data is available locally

## Supporting docs

- `docs/probe/scannet_mesh_first_plan.md`
- `docs/probe/experiment_history.md`
- `docs/probe/experiment_plan.md`
- `docs/probe/todo.md`
### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/autoresearch_probe/configs/phase2_nova_aligned.json`
