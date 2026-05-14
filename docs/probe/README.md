# Probe Docs

## Current canonical status — 2026-05-12

For the latest project state, read:

- `handoff_2026-05-07.md`
- `../../experiments/probe3d/README.md`
- `experiment_plan.md`

Key corrections:

- `scannet_max_interval=1` is now the intended setting because ScanNet preprocessing already uses `frame_skip=20`. Old `max_interval=30` K-view conclusions are invalid/confounded.
- CD-only and two-sample oracle results are not claim-level evidence. Use fixed-sample robust metrics and visual audits.
- The current MLP baseline is mostly a failure-mode baseline: recall is moderate, precision/sharpness are poor.
- The old local `eval_scrream` branch is invalid for claims, but full SCRREAM is now downloaded at `~/datasets/SCRREAM`.
- The active baseline branch is SCRREAM full-data mesh-complete VGGT adapter training on sequence-meta-filtered clean GT.
- The active model-coverage branch is SCRREAM WAN2.1 T2V Route2, which keeps the VGGT ablation's data/GT/split/adapter/decoder/validation fixed and changes only the representation. Slurm job `86307` completed the full 15-run ablation pack; best WAN is `t499/layer09`, above zero/sample-shuffle controls but far below clean-GT VGGT. Treat the route as exploratory / setting-sensitive. Route2.1 `no_noise` / `low_noise` support is now implemented, and the smoke/full-cache/train chain `86342/86343 -> 86350/86351 -> 86357 -> 86358/86359` was queued on `2026-05-12 22:47 CST`.
- Long data generation and training should use `slurm/` scripts with logs in `slurm_out/`.
- Slurm job `86140` completed the 20k / 500k trainplus-test MLP baseline on `air-node-02` with exit `0:0`; `final_metrics.json` reports `best_val_chamfer_l2=0.5149603486061096`.
- Slurm job `86149` completed the robust VGGT layer ablation on `2026-05-08`, but it used the pre-meta-filter GT and is now historical. Clean-GT VGGT ablation job `86286` completed successfully; the current default is layer `16`, with layer `24` as the main comparison point.
- NOVA `scene_n1`, `scene_n2`, `scene_ae`, and VGGT weights are staged under `checkpoints/`; SwanLab is installed in `nova3r`.
- VGGT, Wan2.1, and VidFM3D are Git submodules under `third_party/`; initialize them with `git submodule update --init --recursive`.
- WAN Route2 uses proxy `http://127.0.0.1:17890` through compute-node SSH tunnel logic in `slurm/scrream_wan_t2v_*.sbatch`; non-WAN jobs keep the `7896` proxy default.

This folder records the proposal-facing execution state for the current adapter / decoder experiments.

## Read this first

### 1. What is active now?
The current active formal data/GT line is the **SCRREAM full-data mesh-complete adapter line**.

It is:
- a corrected rerun of the SCRREAM adapter idea using the full dataset at `~/datasets/SCRREAM`
- based on sequence-meta-filtered registered SCRREAM scene meshes, not the invalid `eval_scrream` subset
- constrained to the selected two-input-view union frustum
- launched through Slurm scripts in `slurm/`

Full `.pt` generation is complete for both 10k and 20k target variants. The current formal data file is `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`, which has `train=317`, `val=12`, and `mesh_sequence_meta_filter=True`. The previous 20k / 500k MLP baseline and VGGT layer ablation were run before the sequence-level object filter; use them as historical diagnostics, not final clean-GT claims.

The clean-GT VGGT layer ablation completed as Slurm job `86286`. The current default VGGT layer is `16` (`best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`); layer `24` is the closest comparison point.

The next active experiment branch is **WAN2.1 T2V Route2.1**. It keeps the same SCRREAM `.pt`, mesh-complete GT, split, MLP-L4 adapter, NOVA decoder, and robust validation as the completed Route2 grid, but changes the feature extraction setting first.

WAN Route2 entrypoints:

- dependency pins: `../../experiments/probe3d/requirements-wan-t2v.txt`
- feature cache: `../../experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`
- training: `../../experiments/probe3d/train_wan_t2v_nova_adapter.py`
- Slurm: `../../slurm/scrream_wan_t2v_download.sbatch`, `../../slurm/scrream_wan_t2v_precompute.sbatch`, `../../slurm/scrream_wan_t2v_ablation_pack_train.sbatch`, `../../slurm/scrream_wan_t2v_route21_pack_train.sbatch`

Status on `2026-05-12 16:32 CST`: the WAN checkpoint is present under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`. Full WAN feature precompute completed as jobs `86292`, `86293`, and `86294`, writing `4937` `.pt` files / about `45G`. Training pack job `86307` completed on `air-node-04`, exit `0:0`, elapsed `13:36:04`.

Route2 result:

- best WAN: `t499/layer09`, `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`
- clean-GT VGGT layer `16`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- Route2.1 implementation state: layers `9,14,29` with `no_noise` and `low_noise` cache modes are queued through `86342/86343 -> 86350/86351 -> 86357 -> 86358/86359`; after those complete, run one targeted `t499/layer09 + norm` setting
- deferred WAN settings: `pair_tiled81` vs `ctx81`, I2V/FLF2V conditioning, and broader normalization / adapter sweeps

### 2. What about ScanNet?
The ScanNet v2 mesh-first line remains the diagnostic baseline.

It is:
- a NOVA3R-style extension / transfer probe
- based on reliable mesh-first complete-GT supervision
- useful for understanding failure modes and metric reliability

It is **not** a literal reproduction of official NOVA3R training on `3D-FRONT + ScanNet++V2`.

### 3. What about old SCRREAM?
The older local `eval_scrream` experiments are treated as **invalid for formal claims**, because they used only the released eval subset rather than the official full-data setup.

### 4. What about InteriorGS?
InteriorGS remains a deferred high-quality data option. It is no longer the immediate next training branch after the full SCRREAM dataset became available locally.


## Core documents

- `handoff_2026-05-03.md`
  - historical machine handoff for the first SCRREAM full mesh-complete prep chain

- `handoff_2026-05-07.md`
  - current SCRREAM data / Slurm / submodule handoff

- `scannet_mesh_first_plan.md`
  - the current formal ScanNet v2 plan, assumptions, implementation status, and launch semantics

- `experiment_history.md`
  - the honest history, including corrections and invalidated branches

- `experiment_plan.md`
  - phased execution plan from the current state forward

- `interiorgs_training_plan.md`
  - deferred high-quality indoor dataset migration path

- `todo.md`
  - current actionable task list

## Current ScanNet probe baseline

The current ScanNet branch has shifted from a long formal MLP run to a short probe loop, because that isolated the failure mode faster.

Current best numeric baseline:

- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- objective: direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best validation CD: `0.08745259`
- output dir: `experiments/probe3d/result/probe_trials/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

Interpretation:

- direct Chamfer fixed a large train/eval objective mismatch compared with `nova_flow`
- the result is still not visually clean: prediction recall is reasonable, but precision / outlier control is poor
- the next project direction is to test whether corrected SCRREAM full mesh-complete supervision gives a cleaner training signal than the current ScanNet transfer setup

### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json`
