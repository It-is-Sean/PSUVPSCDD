#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17}
NOVA_CKPT=${NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}
TORCHRUN_BIN=${TORCHRUN_BIN:-/data1/jcd_data/miniconda3/envs/nova3r/bin/torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29602}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/probe3d/result/scannet_ca_adapter_l4_h512_lr2e-5_seed17}

"$TORCHRUN_BIN" --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
  experiments/probe3d/train_vggt_nova_cross_attention_adapter.py \
  --dataset scannet \
  --data_root "$DATA_ROOT" \
  --train_split train \
  --val_split val \
  --test_split test \
  --final_test \
  --eval_batches 0 \
  --test_eval_batches 0 \
  --nova_ckpt "$NOVA_CKPT" \
  --adapter_layers 4 \
  --adapter_hidden_dim 512 \
  --attention_heads 8 \
  --attention_mlp_ratio 2.0 \
  --lr 2e-5 \
  --batch_size 1 \
  --num_workers 4 \
  --max_steps 3000 \
  --save_every 500 \
  --val_every 500 \
  --num_views 4 \
  --num_queries 2048 \
  --save_ply_queries 40960 \
  --amp \
  --wandb \
  --wandb_project PSUVPSC3DD \
  --wandb_entity 3dprobe \
  --wandb_name "$(basename "$OUTPUT_DIR")" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
