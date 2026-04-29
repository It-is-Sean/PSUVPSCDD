# NOVA3R 3D Probe

## Current status override — 2026-04-29 afternoon

The active probe state is summarized in `autoresearch_probe/CURRENT_STATE.md`. Important current constraints:

- use corrected `scannet_max_interval=1` for the frame_skip=20 ScanNet data;
- treat old `max_interval=30` runs as confounded;
- do not interpret single symmetric CD or two-sample oracle averages as claim-level evidence;
- use fixed-sample robust eval / videos before launching new adapter claims;
- AutoResearchClaw is integrated under `../../researchclaw/` and supervised every 15 minutes for proposal alignment.


Minimal collaborator-side probing experiment for decoding complete 3D geometry from frozen NOVA3R / VGGT features.

This repo now also contains the more structured `docs/probe/`, `configs/probe/`, `nova3r/probe/`, and `scripts/probe/` workspace. The files in `experiments/probe3d/` are kept as the direct experimental path.

The backbone is frozen. Only the lightweight adapter / decoder heads are trained.

## Critical data warning

Do **not** treat the local `eval_scrream` package as training data for formal SCRREAM experiments.

That package is the released **evaluation subset** (~1.6 GB on the current machine), not the official full SCRREAM dataset (~200 GB scale). Historical runs that trained from `eval_scrream` should be treated as **debug / invalid-for-claim** runs, not final evidence.

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
## Current ScanNet autoresearch-probe status

The current best short-run ScanNet result is not from the old long formal MLP schedule, but from the autoresearch harness:

- harness: `experiments/probe3d/autoresearch_probe/`
- best run: `p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`
- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- loss: direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best validation CD: `0.08745259`

Interpretation: direct rollout Chamfer fixed the large train/eval objective mismatch seen with `nova_flow`, but the visual point cloud is still loose / thick / outlier-heavy. Future runs should optimize precision / sharpness, not just symmetric CD.
### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/autoresearch_probe/configs/phase2_nova_aligned.json`
