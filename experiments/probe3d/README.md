# NOVA3R 3D Probe

Minimal collaborator-side probing experiment for decoding complete 3D geometry from frozen NOVA3R / VGGT features.

This repo now also contains the more structured `docs/probe/`, `configs/probe/`, `nova3r/probe/`, and `scripts/probe/` workspace. The files in `experiments/probe3d/` are kept as the direct experimental path.

The backbone is frozen. Only the lightweight adapter / decoder heads are trained.

## Step 1: Build Adapter Data

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/prepare_scrream_adapter_data.py \
  --data_root /path/to/eval_scrream \
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
