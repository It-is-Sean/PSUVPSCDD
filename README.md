# PSUVPSC3DD / Probe Workspace

## Current canonical status — 2026-05-12

This branch is now a **server-side research workspace**. The source of truth is:

- `PROPOSAL.md`
- `PROJECT.md`
- `experiments/probe3d/README.md`
- `docs/probe/handoff_2026-05-07.md`
- `docs/probe/handoff_2026-05-03.md`
- `docs/probe/experiment_plan.md`

Important corrections that override older sections below:

1. **ScanNet view interval:** processed data already uses `frame_skip=20`; corrected experiments use `scannet_max_interval=1` (adjacent processed frames, roughly 20 raw frames). Older `max_interval=30` runs are interval-confounded.
2. **Metric reliability:** single symmetric CD and two-sample oracle averages are diagnostic only. Current comparisons use fixed samples, pred-to-GT precision, GT-to-pred recall, F-score thresholds, trimmed CD, and representative renders.
3. **Current MLP baseline:** K2/interval=1 `anchor_frustum + MLP-L4 + chamfer_sample` is recall-heavy but precision/outlier-poor on fixed-30 robust eval: F@0.05 mean/median `0.291/0.275`, precision@0.05 mean `0.204`, recall@0.05 mean `0.532`.
4. **Latest structured-adapter check:** K2/interval=1 `anchor_frustum + cross_attention L2/H512 + chamfer_sample` completed 1000 steps with validation CD `0.54222615`, which is not better than the MLP baseline by scalar validation CD. Its fixed-30 robust eval should be inspected before any claim if result artifacts are available.
5. **SCRREAM full-data status:** the old `eval_scrream` package is still invalid for claims, but the full SCRREAM tree is now available locally at `~/datasets/SCRREAM`.
6. **Current SCRREAM GT path:** the active data bridge uses registered sequence-filtered scene meshes as the complete target source. For `mesh_complete`, it reads each SCRREAM sequence `meta.txt`, samples only the listed object meshes from `sceneXX/meshes/*.obj` proportional to surface area, crops to the selected two-view union frustum, transforms targets into the first input camera frame, and exports fixed-size adapter targets.
7. **Slurm convention:** long data generation and training jobs should be launched through scripts in `slurm/`, with logs in `slurm_out/`.
8. **SCRREAM baseline status:** job `86140` completed the old 20k/500k SCRREAM mesh-complete MLP baseline, and job `86149` completed the pre-meta-filter robust VGGT layer ablation. A sequence-meta-filtered clean GT was regenerated on `2026-05-10`; clean-GT VGGT job `86286` completed successfully on `2026-05-11`. The current clean-GT default is VGGT layer `16`, with layer `24` as the main comparison point.
9. **Local weights:** NOVA3R `scene_n1`, `scene_n2`, `scene_ae`, and VGGT weights are staged under `checkpoints/`; non-WAN Slurm scripts default network proxy variables to `http://127.0.0.1:7896`.
10. **Third-party source:** VGGT, Wan2.1, and VidFM3D are Git submodules under `third_party/`; run `git submodule update --init --recursive` after a fresh clone. Wan2.1 / WAN probe dependencies stay separate from the root env.
11. **WAN Route2 status:** the active representation probe is WAN2.1 T2V video-context features on the same clean SCRREAM `.pt`, split, MLP adapter, NOVA decoder, and robust validation setup as the VGGT ablation. Slurm job `86307` completed the full 15-run ablation pack successfully; best WAN is `t499/layer09` with `F@0.10=0.46988987902779306`, which is above zero/sample-shuffle controls but far below clean-GT VGGT layer `16`. Treat the branch as exploratory and likely setting-sensitive. Route2.1 is now implemented for `no_noise` / `low_noise` caches on layers `9,14,29`; the queued execution chain on `2026-05-12 22:47 CST` is `86342/86343 -> 86350/86351 -> 86357 -> 86358/86359`. A targeted `t499/layer09 + norm` setting remains after that. WAN repo/checkpoint jobs use proxy `http://127.0.0.1:17890` through the Slurm SSH tunnel logic in `slurm/scrream_wan_t2v_*.sbatch`.

This repository is currently a **research execution workspace** around a simple question:

> can frozen visual backbones (currently VGGT-first) drive a NOVA3R-style decoder through a lightweight adapter, and learn useful complete 3D reconstruction behavior under reliable supervision?

It contains:

- the NOVA3R / DUST3R style reconstruction stack used here as the decoder/data backbone
- the probe / adapter experiments under `experiments/probe3d/`
- proposal-facing documentation under `docs/probe/`

## Current experimental status

There are now four lines, and they should be interpreted differently.

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
- GT points are sampled from `sceneXX/meshes/*.obj` after filtering by the current sequence's `meta.txt`
- mesh sampling is surface-area proportional, so walls / room structure are not underweighted relative to small objects
- targets are cropped to the union frustum of the two input views
- final `target_points` are stored in the first input camera coordinate frame
- the output `.pt` is consumed through `--dataset scrream_adapter --data_root <adapter.pt>`
- for `nova_flow`, SCRREAM targets are normalized with the NOVA `scene_ae` checkpoint `norm_mode`; the local `scene_ae` config reports `median_3`

Current data / job state recorded on 2026-05-12 16:32 CST:

- 10k original adapter data exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt`
  - shape `[329, 10000, 3]`, split `train=223`, `val=12`, `test=94`
- 10k trainplus-test data exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt`
  - split `train=317`, `val=12`
- 20k / 500k data exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000.pt`
  - shape `[329, 20000, 3]`, split `train=223`, `val=12`, `test=94`
- current formal training input exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
  - shape `[329, 20000, 3]`, split `train=317`, `val=12`, `mesh_sequence_meta_filter=True`
  - regenerated by Slurm jobs `86283` and `86284` after deleting old scene-level mesh cache; old contaminated `.pt` files were moved to `experiments/probe3d/adapter_data/deprecated_meta_filter_bug/`
- the sequence-meta filter removes scene-level phantom objects that are not listed in a sequence `meta.txt`; for example `scene08/scene08_reduced_00_000220_000260` now selects 22 mesh files and excludes the three mannequin OBJ files
- `86140` / `scrream_mesh_mlp`: `COMPLETED` on `air-node-02`
  - output: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`
  - SwanLab: `https://swanlab.cn/@JiachengDong/PSUVPSC3DD/runs/eoiupi3bv2g11dvd21ypm`
  - final metrics: first loss `1.4599288702011108`, final loss `0.8398033976554871`, best train loss `0.5998285412788391`, best validation Chamfer-L2 `0.5149603486061096`
- `86149` / `scrream_vggt_layer_pack`: `COMPLETED` on `2026-05-08`, but it used the pre-meta-filter GT and is now historical rather than the clean-GT baseline
- `86286` / `scrream_vggt_layer_pack`: `COMPLETED` on `air-node-04`, exit `0:0`, elapsed `02:23:11`; output prefix is `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_vggt_layerXX`
- clean-GT layer ranking now uses robust validation, not the old pre-meta-filter result:
  - layer `16`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
  - layer `24`: `best_val_fscore_tau_0.10=0.6802513448244976`, `best_val_pred_to_gt_p90=0.25152526050806046`, `best_val_chamfer_l2=0.03554012098660072`
  - layer `20`: `best_val_fscore_tau_0.10=0.548158719135383`, `best_val_pred_to_gt_p90=0.36711231619119644`, `best_val_chamfer_l2=0.0767408860847354`

### 2. WAN2.1 T2V Route2 line

The WAN Route2 branch is the active model-coverage expansion after the clean-GT VGGT rerun. The old completed VGGT ablation is pre-meta-filter history, so WAN comparisons should use the clean-GT layer `16` default and layer `24` comparison point. WAN should be read as a **video-context representation probe**, not a pair-only image backbone probe.

Fixed setting:

- adapter data: `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- train / val split: `317 / 12`
- GT: SCRREAM registered mesh-complete targets, union-frustum cropped, first input camera frame, `20000` points
- adapter / decoder / validation: same MLP-L4, NOVA `scene_ae`, `nova_flow`, and robust validation path as the VGGT layer ablation
- WAN feature cache: 81-frame RGB windows around each pair, empty prompt, model `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- ablation grid: timesteps `249,499,749` x code layers `9,14,19,24,29`

Implementation entrypoints:

- dependency pins: `experiments/probe3d/requirements-wan-t2v.txt`
- feature cache: `experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`
- training: `experiments/probe3d/train_wan_t2v_nova_adapter.py`
- Slurm: `slurm/scrream_wan_t2v_download.sbatch`, `slurm/scrream_wan_t2v_precompute.sbatch`, `slurm/scrream_wan_t2v_ablation_pack_train.sbatch`

Status on `2026-05-12 16:32 CST`: the WAN checkpoint is downloaded under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers` (~27 GB). Full feature precompute completed as jobs `86292`, `86293`, and `86294`, writing `4937` `.pt` files / about `45G` under `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81`. The first full training attempt `86306` failed immediately on a reduced-loss float `.item()` bug; the training script was patched, and replacement pack job `86307` completed all 15 runs on `air-node-04`, exit `0:0`, elapsed `13:36:04`.

WAN Route2 result:

- best WAN: `t499/layer09`, `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`
- clean-GT VGGT layer `16`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- interpretation: WAN Route2 is live but setting-limited; current T2V noisy-denoising hidden states are not competitive geometry features under the fixed adapter setting
- next audit already queued: Route2.1 `no_noise` / `low_noise` for layers `9,14,29`, via smoke/full-cache/train chain `86342/86343 -> 86350/86351 -> 86357 -> 86358/86359`; after that run targeted `t499/layer09 + norm`
- defer `pair_tiled81`, I2V/FLF2V, and broader normalization / adapter sweeps

### 3. ScanNet v2 line

The ScanNet v2 mesh-first extension remains the current diagnostic baseline.

Important interpretation note:

- this is a **NOVA3R-style extension / transfer probe** on ScanNet v2
- it is **not** a literal reproduction of the official NOVA3R scene-training recipe, which is described around `3D-FRONT + ScanNet++V2`

The training structure remains aligned with the proposal direction:

- frozen visual backbone
- lightweight adapter
- NOVA3R-style decoder / flow-matching training path
- reliable **complete-GT style** supervision

### 4. InteriorGS line

InteriorGS remains a plausible future data-quality migration path, but it is no longer the immediate next step. The current priority is to inspect the completed SCRREAM full mesh-complete 20k / 500k MLP baseline before choosing the next data-quality migration or adapter branch.

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
2. use clean-GT VGGT layer `16` as the current default representation, with layer `24` as the closest comparison point
3. monitor and evaluate the queued WAN Route2.1 chain: `no_noise` / `low_noise` cache modes for layers `9,14,29`, then run the targeted `t499/layer09 + norm` training setting
4. expand SCRREAM training sample scale beyond the current 329 official pairs after the Route2.1 audit
5. keep the fixed-30 ScanNet metrics as a failure-mode baseline

## Documentation map

- `AGENTS.md` — project-level instructions for future coding agents
- `PROJECT.md` — current project-level status and next steps
- `docs/probe/README.md` — probe-doc entry point
- `docs/probe/handoff_2026-05-07.md` — current SCRREAM data / Slurm / submodule handoff
- `docs/probe/handoff_2026-05-03.md` — historical machine handoff for the first SCRREAM mesh-complete prep chain
- `docs/probe/scannet_mesh_first_plan.md` — current ScanNet formal plan
- `docs/probe/experiment_history.md` — what really happened, including corrections
- `docs/probe/experiment_plan.md` — phased execution plan from here
- `docs/probe/interiorgs_training_plan.md` — deferred high-quality dataset migration plan
- `docs/probe/todo.md` — current actionable task list
- `slurm/` — Slurm job scripts; logs should go to `slurm_out/`
