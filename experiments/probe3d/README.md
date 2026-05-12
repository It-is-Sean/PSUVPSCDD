# NOVA3R 3D Probe

## Current status override — 2026-05-12

The active probe state has moved to the corrected full SCRREAM branch. Important current constraints:

- use corrected `scannet_max_interval=1` for the frame_skip=20 ScanNet data;
- treat old `max_interval=30` runs as confounded;
- do not interpret single symmetric CD or two-sample oracle averages as claim-level evidence;
- use fixed-sample robust eval / videos before launching new adapter claims;
- treat old `eval_scrream` runs as invalid for claims because they used the released eval subset;
- use the downloaded full SCRREAM tree at `~/datasets/SCRREAM` for the corrected branch;
- default SCRREAM full-data target source is now sequence-filtered registered mesh-complete, not dense depth aggregation;
- submit long data-prep and training work through `slurm/`; Slurm logs go to `slurm_out/`.
- job `86140` completed the pre-meta-filter 20k / 500k trainplus-test MLP baseline on `air-node-02` with exit `0:0`; treat it as historical after the sequence-meta GT correction.
- job `86286` completed the VGGT layer ablation on clean sequence-meta-filtered GT under `*robustval_metafilter_seed17_vggt_layerXX`; VGGT layer `16` is the current default, with layer `24` as the main comparison point.
- WAN sanity job `86316` completed; WAN pack job `86307` completed all 15 Route2 runs. Best WAN is `t499/layer09`, above zero/sample-shuffle controls but much weaker than clean-GT VGGT, so treat the branch as exploratory and setting-sensitive.
- initialize third-party submodules with `git submodule update --init --recursive`; VGGT training imports from `third_party/vggt`.


Minimal collaborator-side probing experiment for decoding complete 3D geometry from frozen NOVA3R / VGGT features.

This repo now also contains the more structured `docs/probe/`, `configs/probe/`, `nova3r/probe/`, and `scripts/probe/` workspace. The files in `experiments/probe3d/` are kept as the direct experimental path.

The backbone is frozen. Only the lightweight adapter / decoder heads are trained.

## Critical data warning

Do **not** treat the local `eval_scrream` package as training data for formal SCRREAM experiments.

That package is the released **evaluation subset** (~1.6 GB on the current machine), not the official full SCRREAM dataset (~200 GB scale). Historical runs that trained from `eval_scrream` should be treated as **debug / invalid-for-claim** runs, not final evidence.

## SCRREAM-full mesh-complete adapter bridge

For the full SCRREAM layout at `~/datasets/SCRREAM`, use `prepare_scrream_full_adapter_data.py` with `--target_source mesh_complete`. This path reads the official two-view pair list, uses the two RGB frames as adapter inputs, reads the current sequence `meta.txt`, samples only the listed registered scene meshes, crops the complete point cloud to the selected input-pair union frustum, and stores fixed-size target point clouds in the first input view coordinate frame.

Current target semantics:

- pair list: `data/scrream/scrream_n2_list.json`
- input images: the two pair frames, for example `scene09/scene09_full_00 200 275`
- mesh source: `sceneXX/meshes/*.obj` filtered by the current sequence `meta.txt`
- mesh sampling: surface-area proportional across the selected sequence object meshes
- cache: sequence/object-set mesh reservoirs under `experiments/probe3d/adapter_data/mesh_cache/`
- frustum crop: keep mesh points inside at least one selected input-view frustum with positive depth
- target frame: first input camera coordinates
- final target: deterministic FPS to the requested target count, with replacement padding only when the crop is undersized
- output schema: `scene_ids`, `target_points`, `splits`, `metadata`, and global `meta`

The earlier equal-points-per-OBJ preview under `scrream_mesh_complete_n2_preview/` underweighted room-scale surfaces. The accepted current preview uses area-proportional sampling and lives under:

- `experiments/probe3d/adapter_data/scrream_mesh_complete_area_n2_preview/`

## SCRREAM WAN-T2V video-context probe

The WAN route2 probe keeps the current SCRREAM N2 adapter data and mesh-complete GT unchanged, but replaces VGGT features with precomputed WAN2.1 T2V features. This is a **video-context** representation: each two-view pair is embedded by loading WAN T2V on an 81-frame RGB window around the pair, then keeping only the two temporal slices corresponding to the original pair frames. Adapter training reads the cached WAN features and does not run WAN inside every training step.

Route2 defaults:

- model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- local checkpoint: `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`
- timesteps: `249,499,749`
- code layer ids: `9,14,19,24,29` (human layers 10/15/20/25/30)
- cache shape per sample: `[3120,1536]`
- network proxy for repo/checkpoint fetches: `http://127.0.0.1:17890`

On this Slurm cluster the `17890` proxy is bound to login-node loopback. WAN Slurm jobs therefore open an SSH local tunnel from the compute node back to `air-server:127.0.0.1:17890`, then export `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` to the compute node's local forwarded port. Override with `WAN_PROXY_SSH_HOST`, `WAN_PROXY_REMOTE_PORT`, `WAN_PROXY_LOCAL_PORT`, or set `WAN_DISABLE_PROXY=1`.

WAN-specific Python dependencies are isolated in `experiments/probe3d/requirements-wan-t2v.txt`. They are not merged into the root environment files. Job `slurm/scrream_wan_t2v_download.sbatch` installs/validates them by default before downloading the checkpoint.

Status on `2026-05-12 16:32 CST`: the WAN checkpoint is present under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers` (~27 GB). Full feature precompute completed as jobs `86292`, `86293`, and `86294`; cache root `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81` contains `4937` `.pt` files / about `45G`. The first full training pack `86306` failed immediately on a final-loss scalar bug, `train_wan_t2v_nova_adapter.py` was patched, and replacement pack job `86307` completed all 15 runs on `air-node-04`, exit `0:0`, elapsed `13:36:04`.

Route2 result:

- best WAN: `t499/layer09`, `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`
- clean-GT VGGT layer `16`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- next Route2.1 audit: add `no_noise` / `low_noise` WAN cache modes for layers `9,14,29`, plus a targeted `t499/layer09 + norm` setting
- deferred WAN settings: `pair_tiled81` vs `ctx81`, I2V/FLF2V conditioning, and broader feature-normalization / adapter sweeps

Download/preflight WAN dependencies and checkpoint:

```bash
sbatch slurm/scrream_wan_t2v_download.sbatch
```

Validate all SCRREAM pair windows without loading WAN:

```bash
WAN_WINDOW_ONLY=1 sbatch slurm/scrream_wan_t2v_precompute.sbatch
```

Feature smoke:

```bash
WAN_TIMESTEPS=749 WAN_LAYERS=20 WAN_MAX_SAMPLES=2 \
  sbatch slurm/scrream_wan_t2v_precompute.sbatch
```

Full feature cache:

```bash
sbatch slurm/scrream_wan_t2v_precompute.sbatch
```

Full 15-run adapter ablation:

```bash
sbatch slurm/scrream_wan_t2v_ablation_pack_train.sbatch
```

Run these commands from the `nova3r` conda environment.

Small smoke directly from shell if you only need a quick local check:

```bash
python3 experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py \
  --data_root ~/datasets/SCRREAM \
  --pair_list data/scrream/scrream_n2_list.json \
  --output_path experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke.pt \
  --target_source mesh_complete \
  --target_points 10000 \
  --max_samples 2 \
  --save_preview_dir experiments/probe3d/adapter_data/scrream_mesh_complete_n2_preview
```

Preferred Slurm smoke:

```bash
SCRREAM_ADAPTER_OUT=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_EXTRA_ARGS="--max_samples 8 --save_preview_dir experiments/probe3d/adapter_data/scrream_mesh_complete_smoke8_preview" \
sbatch slurm/scrream_mesh_complete_prepare.sbatch
```

The prep script runs `check_scrream_adapter_training_inputs.py` after writing the `.pt`, so schema / split / frame-path issues fail before training.

Full adapter dataset after the smoke preview looks correct:

```bash
sbatch slurm/scrream_mesh_complete_prepare.sbatch
```

Generated formal datasets now available locally:

- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt`
  - shape `[329, 10000, 3]`
  - split `train=223`, `val=12`, `test=94`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt`
  - split `train=317`, `val=12`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000.pt`
  - shape `[329, 20000, 3]`
  - `mesh_sample_points=500000`
  - split `train=223`, `val=12`, `test=94`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
  - split `train=317`, `val=12`, `mesh_sequence_meta_filter=True`

The `trainplus_test` files rewrite split labels only: `test` becomes `train`, and `val` is preserved. They have no held-out test split.

The 20k / 500k data was regenerated on `2026-05-10` with sequence-level object filtering from each sequence `meta.txt`. Old pre-meta-filter `.pt` files were moved to `experiments/probe3d/adapter_data/deprecated_meta_filter_bug/`. The corresponding mesh cache now uses sequence/meta-hash labels such as `scene08_scene08_reduced_00_meta7c4e1a9bf084_...npz`, so it no longer reuses scene-level all-object reservoirs.

Training smoke after a smoke `.pt` exists:

```bash
SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_MAX_STEPS=20 \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/scrream_mesh_complete_smoke8_mlp_smoke \
sbatch slurm/scrream_mesh_complete_mlp_smoke.sbatch
```

The smoke script uses the same training entrypoint and loss as the full run. It checks the adapter `.pt`, loads the NOVA scene autoencoder checkpoint, extracts VGGT features, verifies adapter gradients, and writes a short checkpoint / PLY export.

First full MLP baseline after the full `.pt` exists:

```bash
sbatch slurm/scrream_mesh_complete_mlp_train.sbatch
```

Current 20k / 500k trainplus-test run:

```bash
SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17 \
SCRREAM_FEATURE_CACHE_DIR=experiments/probe3d/feature_cache/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_vggt23 \
SCRREAM_SWANLAB=1 \
SCRREAM_NUM_QUERIES=20000 \
SCRREAM_MAX_STEPS=9510 \
SCRREAM_SAVE_EVERY=500 \
SCRREAM_VAL_EVERY=500 \
SCRREAM_EVAL_BATCHES=2 \
sbatch --qos=high --nodelist=air-node-02 --gres=gpu:a100:1 --mem=32G \
  slurm/scrream_mesh_complete_mlp_train.sbatch
```

This submitted job `86140`, which completed on `air-node-02` with exit `0:0`, elapsed `00:26:26`, and Slurm end time `2026-05-07 22:40:22 CST`.

Important: job `86140` and the old robust VGGT layer ablation job `86149` used the pre-meta-filter GT. The clean-GT rerun is completed Slurm job `86286`, with output prefix:

- `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_vggt_layerXX`

Current clean-GT VGGT ranking:

- layer `16`: current default, `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- layer `24`: main comparison point, `best_val_fscore_tau_0.10=0.6802513448244976`, `best_val_pred_to_gt_p90=0.25152526050806046`, `best_val_chamfer_l2=0.03554012098660072`
- layer `20`: no longer the clean-GT default, `best_val_fscore_tau_0.10=0.548158719135383`

Completed run artifacts:

- output: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`
- SwanLab: `https://swanlab.cn/@JiachengDong/PSUVPSC3DD/runs/eoiupi3bv2g11dvd21ypm`
- `final_metrics.json`: `first_loss=1.4599288702011108`, `final_loss=0.8398033976554871`, `best_loss=0.5998285412788391`, `best_val_chamfer_l2=0.5149603486061096`
- latest `validation_metrics.json`: step `9500`, `val_chamfer_l2=0.6779176592826843`

Future multi-GPU follow-up runs should use the script's DDP path:

```bash
SCRREAM_GPUS_PER_NODE=4 \
SCRREAM_EPOCHS=50 \
SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/<new-run-name> \
SCRREAM_FEATURE_CACHE_DIR=experiments/probe3d/feature_cache/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_vggt23 \
SCRREAM_NUM_QUERIES=20000 \
sbatch --qos=high --nodelist=<node-with-free-a100s> --gres=gpu:a100:4 --mem=96G \
  slurm/scrream_mesh_complete_mlp_train.sbatch
```

When `SCRREAM_GPUS_PER_NODE>1`, the Slurm script launches `torchrun --standalone`; with `SCRREAM_EPOCHS` it computes the correct distributed step count from `train_count / (batch_size_per_gpu * gpu_count)`.

If checkpoints are not in the default repo paths, pass them as environment variables:

```bash
SCRREAM_NOVA_CKPT=/path/to/scene_ae/checkpoint-last.pth \
SCRREAM_VGGT_WEIGHTS=/path/to/VGGT-1B/model.pt \
sbatch slurm/scrream_mesh_complete_mlp_train.sbatch
```

Training defaults follow the current NOVA-style adapter baseline:

- `--dataset scrream_adapter`
- `--num_views 2`
- `--adapter_type mlp`
- `--adapter_layers 4`
- `--adapter_hidden_dim 1024`
- `--loss_type nova_flow`
- `--num_queries 10000`
- persistent VGGT layer-23 cache under `experiments/probe3d/feature_cache/`
- target normalization follows the NOVA `scene_ae` decoder checkpoint `norm_mode`; local `scene_ae` currently reports `median_3`
- full MLP Slurm training enables SwanLab by default with project `PSUVPSC3DD`

Local weights currently staged under `checkpoints/`:

- `scene_n1/checkpoint-last.pth`
- `scene_n2/checkpoint-last.pth`
- `scene_ae/checkpoint-last.pth`
- `vggt/model.pt`

The Slurm scripts default `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` to `http://127.0.0.1:7896` for network-dependent HuggingFace / SwanLab access.

The older `prepare_scrream_adapter_data.py` remains a legacy LDI / pseudo-GT path. It expects `ldi/` and `*_ldi.npz`, so it is not the right entrypoint for the downloaded full SCRREAM tree.

### Historical depth-GT / depth-mix branches

`prepare_scrream_full_adapter_data.py` still supports `--target_source depth_gt_dense` as an alternate baseline. That mode aggregates `depth_gt` frames between the input pair, voxel-filters duplicates, crops to the input union frustum, and stores targets in the first input camera frame.

Use it only when intentionally comparing depth aggregation against mesh-complete targets. It does not produce truly mesh-complete surfaces; it can include surfaces seen by nearby dense frames that were not visible in the first input, but it cannot recover unobserved mesh backsides except where other depth frames observed them.

Local `scrream_official_depth_mix_*` `.pt` and preview artifacts under `experiments/probe3d/adapter_data/` are historical ablations. They are not part of the current adapter-training line and should be ignored unless the user explicitly asks for a depth-mix ablation.

## Legacy LDI / pseudo-GT adapter path

The following path is kept for historical reference and only applies to SCRREAM-style data that has `ldi/` and `*_ldi.npz` files. It is not the path for `~/datasets/SCRREAM`.

## Step 1: Build Adapter Data

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/prepare_scrream_adapter_data.py \
  --data_root /path/to/official_full_scrream \
  --output_root experiments/probe3d/adapter_data \
  --group_size 4 \
  --sample_stride 1 \
  --pad_short_scenes \
  --pseudo_gt_views 2 \
  --pseudo_gt_queries 20000 \
  --feature_ckpt checkpoints/scene_n2/checkpoint-last.pth \
  --pseudo_gt_ckpt checkpoints/scene_n2/checkpoint-last.pth \
  --skip_failures
```

This creates:

- a scene-level `train` / `val` / `test` split
- 4-frame groups per scene
- a manifest JSON with per-sample frame lists
- an adapter-training `.pt` dataset with `features`, `target_points`, `splits`, and per-sample `metadata`

`pseudo_gt_views=2` keeps pseudo GT on the official NOVA3R inference path while each adapter sample still carries 4 frames.

Before running this step, verify that `--data_root` points to the **official full SCRREAM release** and contains the expected full-scene metadata (for example `camera_pose/` where required). If it only looks like `eval_scrream`, stop and fix the data source first.

## Step 2: Run Official Baseline

Run the official NOVA3R SCRREAM baseline first to verify checkpoints, config, dataset paths, and CUDA environment.

## Step 3: Inspect Outputs

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/inspect_nova3r_outputs.py \
  +experiment.ckpt_path=checkpoints/scene_n1/checkpoint-last.pth \
  +experiment.test_dataset_name=scrream_n1 \
  +experiment.data_root=/path/to/datasets
```

The script prints output keys and tensor shapes. If model config is not found, run it with the same Hydra model config used by `eval/mv_recon/test_nova3r.py`.

## Step 4: Extract Features

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/extract_nova3r_features.py \
  +experiment.ckpt_path=checkpoints/scene_n1/checkpoint-last.pth \
  +experiment.test_dataset_name=scrream_n1 \
  +experiment.data_root=/path/to/datasets \
  --output_path experiments/probe3d/features/nova3r_scrream_n1.pt
```

If `--feature_key` is omitted, the script prints candidates and uses the first feature-like tensor with a warning. Use `--target_key` only after inspecting batch tensor shapes if automatic complete-point extraction fails.

## Step 5: Train Probe

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/train_probe.py \
  --feature_path experiments/probe3d/adapter_data/scrream_adapter_dataset.pt \
  --save_path experiments/probe3d/checkpoints/probe_nova3r_d1.pt \
  --train_split train \
  --val_split val \
  --adapter_depth 1 \
  --latent_dim 512 \
  --num_points 8192 \
  --batch_size 4 \
  --epochs 100
```

## Step 6: Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/eval_probe.py \
  --feature_path experiments/probe3d/adapter_data/scrream_adapter_dataset.pt \
  --checkpoint experiments/probe3d/checkpoints/probe_nova3r_d1.pt \
  --split test
```

Use `--save_predictions` to write predicted point clouds to `experiments/probe3d/outputs/`.

## Notes

- Keep `adapter_depth` small. A shallow adapter is the point of this representation probe.
- If `pytorch3d` is unavailable, this code uses a simple `torch.cdist` Chamfer L2 implementation.
- VGGT code is now available in-repo at `third_party/vggt`.
- `dust3r/datasets/` and `datasets_preprocess/` are also vendored locally, so the common data-loading / ScanNet-prep path is less dependent on an external CUT3R checkout.
- For the structured shared-decoding sanity path, see `scripts/probe/run_vggt_to_nova3r_decoder.py` and `docs/probe/`.
- Feature files, checkpoints, outputs, and datasets are ignored by git.
## Current ScanNet probe status

The current best short-run ScanNet result is not from the old long formal MLP schedule, but from the compact probe harness:

- harness: `experiments/probe3d/probe_trials/`
- best run: `p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`
- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- loss: direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best validation CD: `0.08745259`

Interpretation: direct rollout Chamfer fixed the large train/eval objective mismatch seen with `nova_flow`, but the visual point cloud is still loose / thick / outlier-heavy. The next active data task is to inspect the completed SCRREAM full mesh-complete adapter baseline; InteriorGS is deferred until this corrected full-data branch is understood.
### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json`
