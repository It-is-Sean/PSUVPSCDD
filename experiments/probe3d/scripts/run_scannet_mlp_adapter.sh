#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17}
NOVA_CKPT=${NOVA_CKPT:-/home/jcd/.openclaw/workspace/projects/probe/checkpoints/scene_ae/checkpoint-last.pth}
PYTHON_BIN=${PYTHON_BIN:-/data1/jcd_data/miniconda3/envs/nova3r/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/data1/jcd_data/miniconda3/envs/nova3r/bin/torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29601}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/probe3d/result/scannet_mlp_adapter_l4_lr5e-5_seed17_step5000_val10}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-PSUVPSC3DD}
SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_VIEWS=${NUM_VIEWS:-4}
MAX_STEPS=${MAX_STEPS:-5000}
VAL_EVERY=${VAL_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-500}
MAX_VAL_SCENES=${MAX_VAL_SCENES:-10}
MAX_TEST_SCENES=${MAX_TEST_SCENES:-0}
FINAL_TEST=${FINAL_TEST:-0}
EVAL_BATCHES=${EVAL_BATCHES:-0}
TEST_EVAL_BATCHES=${TEST_EVAL_BATCHES:-0}
RESUME=${RESUME:-}

export DATA_ROOT BATCH_SIZE NPROC_PER_NODE NUM_VIEWS

mapfile -t SCHEDULE < <("$PYTHON_BIN" - <<'PY'
import contextlib, math, sys
sys.path.insert(0, '/home/jcd/PSUVPSC3DD_repo')
from experiments.probe3d.vggt_nova_adapter_common_raw import build_scannet_loader
import os
root=os.environ['DATA_ROOT']
batch_size=int(os.environ['BATCH_SIZE'])
world_size=int(os.environ['NPROC_PER_NODE'])
num_views=int(os.environ['NUM_VIEWS'])
with contextlib.redirect_stdout(sys.stderr):
    loader,_=build_scannet_loader(root, batch_size=batch_size, num_workers=0, num_views=num_views, split_override='train')
train_len=len(loader.dataset)
per_rank_samples=math.ceil(train_len/world_size)
steps_per_epoch=math.ceil(per_rank_samples/batch_size)
print(train_len)
print(steps_per_epoch)
PY
)
TRAIN_LEN=${SCHEDULE[0]}
STEPS_PER_EPOCH=${SCHEDULE[1]}

EXTRA_ARGS=()
if [[ -n "$SWANLAB_WORKSPACE" ]]; then
  EXTRA_ARGS+=(--swanlab_workspace "$SWANLAB_WORKSPACE")
fi
if [[ -z "$RESUME" && -f "$OUTPUT_DIR/latest.pth" ]]; then
  RESUME="$OUTPUT_DIR/latest.pth"
fi
if [[ -n "$RESUME" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME")
fi
if [[ "$FINAL_TEST" == "1" || "$FINAL_TEST" == "true" || "$FINAL_TEST" == "TRUE" ]]; then
  EXTRA_ARGS+=(--final_test)
fi

echo "[MLP formal run]"
echo "  data_root=$DATA_ROOT"
echo "  train_len=$TRAIN_LEN"
echo "  steps_per_epoch=$STEPS_PER_EPOCH"
echo "  max_steps=$MAX_STEPS"
echo "  val_every=$VAL_EVERY"
echo "  save_every=$SAVE_EVERY"
echo "  num_views=$NUM_VIEWS"
echo "  max_val_scenes=$MAX_VAL_SCENES"
echo "  final_test=$FINAL_TEST"
echo "  output_dir=$OUTPUT_DIR"
echo "  resume=${RESUME:-<none>}"

"$TORCHRUN_BIN" --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
  experiments/probe3d/train_vggt_nova_adapter.py \
  --dataset scannet \
  --data_root "$DATA_ROOT" \
  --train_split train \
  --val_split val \
  --test_split test \
  --max_val_scenes "$MAX_VAL_SCENES" \
  --max_test_scenes "$MAX_TEST_SCENES" \
  --eval_batches "$EVAL_BATCHES" \
  --test_eval_batches "$TEST_EVAL_BATCHES" \
  --nova_ckpt "$NOVA_CKPT" \
  --adapter_layers 4 \
  --adapter_hidden_dim 1024 \
  --lr 5e-5 \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_steps "$MAX_STEPS" \
  --save_every "$SAVE_EVERY" \
  --val_every "$VAL_EVERY" \
  --num_views "$NUM_VIEWS" \
  --num_queries 2048 \
  --save_ply_queries 40960 \
  --amp \
  --swanlab \
  --swanlab_project "$SWANLAB_PROJECT" \
  --swanlab_experiment "$(basename "$OUTPUT_DIR")" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}" \
  "$@"
