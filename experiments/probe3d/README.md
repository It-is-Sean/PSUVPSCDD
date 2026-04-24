# NOVA3R 3D Probe

Minimal probing experiment for decoding complete 3D geometry from frozen NOVA3R features on SCRREAM n=1.

The backbone is frozen. Only `SmallAdapter` and `PointDecoder` are trained.

## Step 1: Run Official Baseline

Run the official NOVA3R SCRREAM baseline first to verify checkpoints, config, dataset paths, and CUDA environment.

## Step 2: Inspect Outputs

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/inspect_nova3r_outputs.py \
  +experiment.ckpt_path=checkpoints/scene_n1/checkpoint-last.pth \
  +experiment.test_dataset_name=scrream_n1 \
  +experiment.data_root=/home/wdh/nova3r/datasets
```

The script prints output keys and tensor shapes. If model config is not found, run it with the same Hydra model config used by `eval/mv_recon/test_nova3r.py`.

## Step 3: Extract Features

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/extract_nova3r_features.py \
  +experiment.ckpt_path=checkpoints/scene_n1/checkpoint-last.pth \
  +experiment.test_dataset_name=scrream_n1 \
  +experiment.data_root=/home/wdh/nova3r/datasets \
  --output_path experiments/probe3d/features/nova3r_scrream_n1.pt
```

If `--feature_key` is omitted, the script prints candidates and uses the first feature-like tensor with a warning. Use `--target_key` only after inspecting batch tensor shapes if automatic complete-point extraction fails.

## Step 4: Train Probe

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/train_probe.py \
  --feature_path experiments/probe3d/features/nova3r_scrream_n1.pt \
  --save_path experiments/probe3d/checkpoints/probe_nova3r_d1.pt \
  --adapter_depth 1 \
  --latent_dim 512 \
  --num_points 8192 \
  --batch_size 4 \
  --epochs 100
```

## Step 5: Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/eval_probe.py \
  --feature_path experiments/probe3d/features/nova3r_scrream_n1.pt \
  --checkpoint experiments/probe3d/checkpoints/probe_nova3r_d1.pt
```

Use `--save_predictions` to write predicted point clouds to `experiments/probe3d/outputs/`.

## Notes

- Keep `adapter_depth` small. A shallow adapter is the point of this representation probe.
- If `pytorch3d` is unavailable, this code uses a simple `torch.cdist` Chamfer L2 implementation.
- Feature files, checkpoints, outputs, and datasets are ignored by git.

