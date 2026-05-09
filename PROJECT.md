# PROJECT.md

## Current canonical state — 2026-05-07

The active branch is a server-side research workspace for the PSUVPSC3DD probe. The ScanNet experiments remain the diagnostic baseline, but the immediate training branch has moved to **full SCRREAM mesh-complete adapter training** now that the full SCRREAM tree is available at `~/datasets/SCRREAM`.

The current research interpretation is:

- old ScanNet experiments with `max_interval=30` are confounded because processed data already used `frame_skip=20`; use `scannet_max_interval=1` for the intended NOVA3R-style spacing;
- the raw ScanNet GT audit did not show obviously catastrophic data quality;
- oracle/CD rankings were unstable and should not drive claims;
- the interval-corrected MLP baseline has coverage but poor precision/outlier control, so valid progress must improve robust visual-first metrics, not just symmetric CD;
- the first lightweight cross-attention candidate (`p7_k2_i1_anchor_ca_l2_h512_chamfer_step1000`) completed 1000 steps with validation CD `0.54222615`, so it is not an immediate scalar improvement over MLP;
- the old local `eval_scrream` runs remain invalid for claims, but the corrected full SCRREAM branch is now available and should be evaluated with mesh-complete supervision;
- the current SCRREAM target construction is: registered scene meshes -> surface-area-proportional surface sampling -> two-input-view union-frustum crop -> first input camera coordinates -> FPS / pad to fixed target points;
- Slurm is the expected execution path for data generation and training scripts on this machine;
- SCRREAM job `86140` completed the 20k / 500k trainplus-test MLP baseline on `air-node-02` with exit `0:0`, elapsed `00:26:26`, and Slurm end time `2026-05-07 22:40:22 CST`;
- the SCRREAM 20k baseline output is `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`; `final_metrics.json` reports `first_loss=1.4599288702011108`, `final_loss=0.8398033976554871`, `best_loss=0.5998285412788391`, and `best_val_chamfer_l2=0.5149603486061096`;
- local checkpoints now include `checkpoints/scene_n1/checkpoint-last.pth`, `checkpoints/scene_n2/checkpoint-last.pth`, `checkpoints/scene_ae/checkpoint-last.pth`, and `checkpoints/vggt/model.pt`.

The immediate handoff contract is described in `docs/probe/handoff_2026-05-07.md`, `experiments/probe3d/README.md`, and `docs/probe/experiment_plan.md`.

## Project identity

This repo is currently a **research execution workspace** for probing whether strong frozen visual features can be adapted into a NOVA3R-style scene reconstruction decoder with minimal trainable structure.

The present focus is intentionally narrow:

- image / geometry first
- reliable supervision first
- scale validation before method sprawl

## Current status

There are now three branches, with different evidential status.

### A. SCRREAM full-data branch

The earlier local `eval_scrream` experiments were later found to use only the released **evaluation subset**, not the official full SCRREAM training-scale dataset.

Therefore:

- those old SCRREAM numbers are **invalid as formal evidence**
- they remain useful as engineering/debug history only
- the corrected SCRREAM rerun must use the full tree at `~/datasets/SCRREAM`

Current implementation status:

- data bridge: `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`
- primary target source: `--target_source mesh_complete`
- mesh source: `sceneXX/meshes/*.obj`
- pair source: `data/scrream/scrream_n2_list.json`
- target coordinates: first input camera frame
- generated target shapes: `[329, 10000, 3]` and `[329, 20000, 3]`
- loader path: `python experiments/probe3d/train_vggt_nova_adapter.py --dataset scrream_adapter --data_root <adapter.pt>`
- Slurm prep script: `slurm/scrream_mesh_complete_prepare.sbatch`
- Slurm split-merge helper: `slurm/scrream_merge_test_into_train.sbatch`
- Slurm MLP training script: `slurm/scrream_mesh_complete_mlp_train.sbatch`
- current formal adapter input:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
  - shape `[329, 20000, 3]`, split `train=317`, `val=12`
- completed Slurm training job recorded on 2026-05-07:
  - `86140` / `scrream_mesh_mlp`: `COMPLETED`, exit `0:0`, elapsed `00:26:26`, node `air-node-02`
  - output: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`
  - best validation Chamfer-L2: `0.5149603486061096`
- network proxy default in the Slurm scripts: `http://127.0.0.1:7896`
- SwanLab is installed in `nova3r`, and the full MLP script enables SwanLab by default unless `SCRREAM_SWANLAB=0`
- `slurm/scrream_mesh_complete_mlp_train.sbatch` supports single-node DDP with `SCRREAM_GPUS_PER_NODE`; prefer `SCRREAM_EPOCHS` for future multi-GPU runs so step counts scale by effective batch size.
- Local `scrream_official_depth_mix_*` artifacts are historical ablations and not part of the current training line.

### B. ScanNet v2 branch

The ScanNet v2 mesh-first extension is now the diagnostic baseline.

It should be interpreted as:

- a **NOVA3R-style extension / transfer probe** on ScanNet v2
- not a literal reproduction of official NOVA3R training on `3D-FRONT + ScanNet++V2`

Still, this line is now serious enough to serve as the proposal’s **reliable-target baseline**.

### C. InteriorGS branch

InteriorGS remains a deferred high-quality data option. Do not treat it as the immediate next training branch until the completed SCRREAM mesh-complete baseline has been inspected and understood.

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

The most informative current run is the short ScanNet probe, not the old long formal MLP schedule.

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
  - `experiments/probe3d/result/probe_trials/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

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

1. inspect job `86140`'s `validation_metrics.json`, `final_metrics.json`, exported PLY previews, and representative outputs before making a scientific claim
2. compare the best checkpoint (`best.pth`) against the latest/final checkpoint if qualitative results diverge from the scalar metrics
3. use the multi-GPU Slurm path for follow-up runs when idle GPUs are available
4. keep InteriorGS as the next data-quality option only after SCRREAM is understood

## Supporting docs

- `docs/probe/handoff_2026-05-07.md`
- `docs/probe/scannet_mesh_first_plan.md`
- `docs/probe/experiment_history.md`
- `docs/probe/experiment_plan.md`
- `docs/probe/interiorgs_training_plan.md`
- `docs/probe/todo.md`
- `experiments/probe3d/README.md`

### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json`
