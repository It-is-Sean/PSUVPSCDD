#!/usr/bin/env bash
#SBATCH --job-name=scrream_mesh_mlp
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/result experiments/probe3d/feature_cache

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

if [[ -f slurm/.secrets.env ]]; then
  set -a
  source slurm/.secrets.env
  set +a
fi

if [[ "${SCRREAM_DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
else
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7896}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7896}"
  export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7896}"
fi

resolve_existing_path() {
  local explicit="$1"
  shift
  if [[ -n "$explicit" ]]; then
    if [[ -f "$explicit" ]]; then
      printf '%s\n' "$explicit"
      return 0
    fi
    echo "Requested path does not exist: $explicit" >&2
    return 1
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

ADAPTER_DATA="${SCRREAM_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt}"
NOVA_CKPT="$(resolve_existing_path "${SCRREAM_NOVA_CKPT:-}" \
  "checkpoints/scene_ae/checkpoint-last.pth" \
  "/data1/jcd_data/checkpoints/scene_ae/checkpoint-last.pth" \
  "/data1/jcd_data/nova3r/checkpoints/scene_ae/checkpoint-last.pth" \
  || true)"
VGGT_WEIGHTS="$(resolve_existing_path "${SCRREAM_VGGT_WEIGHTS:-}" \
  "/data1/jcd_data/cache/models/vggt/VGGT-1B/model.pt" \
  "checkpoints/vggt/model.pt" \
  "artifacts/weights/vggt/VGGT-1B/model.pt" \
  || true)"
OUTPUT_DIR="${SCRREAM_OUTPUT_DIR:-experiments/probe3d/result/scrream_mesh_complete_n2_mlp_l4_nova_flow_seed17}"
FEATURE_CACHE_DIR="${SCRREAM_FEATURE_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_mesh_complete_n2_vggt23}"
VGGT_LAYER="${SCRREAM_VGGT_LAYER:-23}"

detect_gpu_count() {
  if [[ -n "${SCRREAM_GPUS_PER_NODE:-}" ]]; then
    printf '%s\n' "$SCRREAM_GPUS_PER_NODE"
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python - <<'PY'
import os
visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
print(max(1, len(visible)))
PY
    return 0
  fi
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$SLURM_GPUS_ON_NODE"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L 2>/dev/null | wc -l)"
    if [[ "$count" -gt 0 ]]; then
      printf '%s\n' "$count"
      return 0
    fi
  fi
  printf '1\n'
}

GPU_COUNT="$(detect_gpu_count)"
BATCH_SIZE="${SCRREAM_BATCH_SIZE:-1}"
if [[ -z "${SCRREAM_MAX_STEPS:-}" && -n "${SCRREAM_EPOCHS:-}" ]]; then
  MAX_STEPS="$(
    ADAPTER_DATA="$ADAPTER_DATA" \
    TRAIN_SPLIT="${SCRREAM_TRAIN_SPLIT:-train}" \
    SCRREAM_EPOCHS="$SCRREAM_EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    GPU_COUNT="$GPU_COUNT" \
    python - <<'PY'
import math
import os
from pathlib import Path

import torch

payload = torch.load(Path(os.environ["ADAPTER_DATA"]), map_location="cpu")
split = os.environ["TRAIN_SPLIT"]
splits = payload.get("splits", ["train"] * len(payload["scene_ids"]))
train_count = sum(1 for item in splits if item == split)
if train_count <= 0:
    raise SystemExit(f"No samples found for split={split!r}")
epochs = float(os.environ["SCRREAM_EPOCHS"])
batch_size = max(1, int(os.environ["BATCH_SIZE"]))
gpu_count = max(1, int(os.environ["GPU_COUNT"]))
print(max(1, math.ceil(train_count * epochs / (batch_size * gpu_count))))
PY
  )"
else
  MAX_STEPS="${SCRREAM_MAX_STEPS:-3000}"
fi

if [[ ! -f "$ADAPTER_DATA" ]]; then
  echo "Missing SCRREAM adapter dataset: $ADAPTER_DATA" >&2
  echo "Generate it first with: sbatch slurm/scrream_mesh_complete_prepare.sbatch" >&2
  exit 2
fi

if [[ -z "$NOVA_CKPT" ]]; then
  echo "Missing NOVA scene autoencoder checkpoint." >&2
  echo "Set SCRREAM_NOVA_CKPT=/path/to/checkpoint-last.pth or download checkpoints/scene_ae/checkpoint-last.pth." >&2
  exit 2
fi

if [[ -z "$VGGT_WEIGHTS" ]]; then
  echo "Warning: no local VGGT weights found. The training script will try VGGT.from_pretrained(), which may require network access." >&2
fi

mkdir -p "$OUTPUT_DIR" "$FEATURE_CACHE_DIR"

VGGT_ARGS=()
if [[ -n "$VGGT_WEIGHTS" ]]; then
  VGGT_ARGS=(--vggt_weights "$VGGT_WEIGHTS")
fi
VAL_PREVIEW_DIR_ARGS=()
if [[ -n "${SCRREAM_VAL_PREVIEW_DIR:-}" ]]; then
  VAL_PREVIEW_DIR_ARGS=(--val_preview_dir "$SCRREAM_VAL_PREVIEW_DIR")
fi

echo "SCRREAM mesh-complete MLP training"
echo "  adapter_data=$ADAPTER_DATA"
echo "  nova_ckpt=$NOVA_CKPT"
echo "  vggt_weights=${VGGT_WEIGHTS:-<from_pretrained>}"
echo "  output_dir=$OUTPUT_DIR"
echo "  feature_cache_dir=$FEATURE_CACHE_DIR"
echo "  max_steps=$MAX_STEPS"
echo "  loss_type=${SCRREAM_LOSS_TYPE:-nova_flow}"
echo "  num_queries=${SCRREAM_NUM_QUERIES:-10000}"
echo "  vggt_layer=$VGGT_LAYER"
echo "  eval_batches=${SCRREAM_EVAL_BATCHES:-0}"
echo "  val_metric_max_points=${SCRREAM_VAL_METRIC_MAX_POINTS:-20000}"
echo "  val_preview_samples=${SCRREAM_VAL_PREVIEW_SAMPLES:-12}"
echo "  val_preview_queries=${SCRREAM_VAL_PREVIEW_QUERIES:-40960}"
echo "  val_flow_t_bins=${SCRREAM_VAL_FLOW_T_BINS:-10}"
echo "  gpus_per_node=$GPU_COUNT"
echo "  batch_size_per_gpu=$BATCH_SIZE"
if [[ -n "${SCRREAM_EPOCHS:-}" ]]; then
  echo "  epochs=$SCRREAM_EPOCHS"
fi
echo "  num_views=2"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

python experiments/probe3d/scripts/check_scrream_adapter_training_inputs.py \
  --adapter_data "$ADAPTER_DATA" \
  --nova_ckpt "$NOVA_CKPT" \
  --num_queries "${SCRREAM_NUM_QUERIES:-10000}" \
  --split "${SCRREAM_TRAIN_SPLIT:-train}"

read -r -a TRAIN_EXTRA_ARGS <<< "${SCRREAM_TRAIN_EXTRA_ARGS:-}"
TRACKING_ARGS=()
if [[ "${SCRREAM_SWANLAB:-1}" != "0" ]]; then
  TRACKING_ARGS=(
    --swanlab
    --swanlab_project "${SCRREAM_SWANLAB_PROJECT:-PSUVPSC3DD}"
    --swanlab_experiment "${SCRREAM_SWANLAB_EXPERIMENT:-scrream_mesh_complete_n2_mlp_l4_nova_flow_seed17}"
  )
  if [[ -n "${SCRREAM_SWANLAB_WORKSPACE:-}" ]]; then
    TRACKING_ARGS+=(--swanlab_workspace "$SCRREAM_SWANLAB_WORKSPACE")
  fi
fi

TRAIN_ARGS=(
  experiments/probe3d/train_vggt_nova_adapter.py
  --dataset scrream_adapter \
  --data_root "$ADAPTER_DATA" \
  --nova_ckpt "$NOVA_CKPT" \
  --adapter_type mlp \
  --adapter_layers "${SCRREAM_ADAPTER_LAYERS:-4}" \
  --adapter_hidden_dim "${SCRREAM_ADAPTER_HIDDEN_DIM:-1024}" \
  --loss_type "${SCRREAM_LOSS_TYPE:-nova_flow}" \
  --num_queries "${SCRREAM_NUM_QUERIES:-10000}" \
  --vggt_layer "$VGGT_LAYER" \
  --num_views 2 \
  --train_split "${SCRREAM_TRAIN_SPLIT:-train}" \
  --val_split "${SCRREAM_VAL_SPLIT:-val}" \
  --test_split "${SCRREAM_TEST_SPLIT:-test}" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "${SCRREAM_NUM_WORKERS:-4}" \
  --lr "${SCRREAM_LR:-1e-4}" \
  --max_steps "$MAX_STEPS" \
  --save_every "${SCRREAM_SAVE_EVERY:-500}" \
  --val_every "${SCRREAM_VAL_EVERY:-500}" \
  --eval_batches "${SCRREAM_EVAL_BATCHES:-0}" \
  --val_flow_t_bins "${SCRREAM_VAL_FLOW_T_BINS:-10}" \
  --val_metric_max_points "${SCRREAM_VAL_METRIC_MAX_POINTS:-20000}" \
  --val_preview_samples "${SCRREAM_VAL_PREVIEW_SAMPLES:-12}" \
  --val_preview_queries "${SCRREAM_VAL_PREVIEW_QUERIES:-40960}" \
  --save_ply_queries "${SCRREAM_SAVE_PLY_QUERIES:-40960}" \
  --feature_cache_dir "$FEATURE_CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${VAL_PREVIEW_DIR_ARGS[@]}" \
  "${VGGT_ARGS[@]}" \
  "${TRACKING_ARGS[@]}" \
  "${TRAIN_EXTRA_ARGS[@]}"
)

if [[ "$GPU_COUNT" -gt 1 ]]; then
  if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun is required for SCRREAM_GPUS_PER_NODE=$GPU_COUNT but was not found in PATH." >&2
    exit 2
  fi
  echo "Launching DDP with torchrun --nproc_per_node=$GPU_COUNT"
  torchrun --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" "${TRAIN_ARGS[@]}"
else
  echo "Launching single-process training"
  python "${TRAIN_ARGS[@]}"
fi
