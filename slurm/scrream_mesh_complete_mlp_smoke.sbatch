#!/usr/bin/env bash
#SBATCH --job-name=scrream_mesh_smoke
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

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

ADAPTER_DATA="${SCRREAM_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_current_gt_preview.pt}"
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
OUTPUT_DIR="${SCRREAM_OUTPUT_DIR:-experiments/probe3d/result/scrream_mesh_complete_current_gt_preview_mlp_smoke}"
FEATURE_CACHE_DIR="${SCRREAM_FEATURE_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_mesh_complete_smoke_vggt23}"

if [[ ! -f "$ADAPTER_DATA" ]]; then
  echo "Missing smoke adapter dataset: $ADAPTER_DATA" >&2
  echo "Generate a smoke .pt with prepare_scrream_full_adapter_data.py before submitting this job." >&2
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

echo "SCRREAM mesh-complete MLP smoke"
echo "  adapter_data=$ADAPTER_DATA"
echo "  nova_ckpt=$NOVA_CKPT"
echo "  vggt_weights=${VGGT_WEIGHTS:-<from_pretrained>}"
echo "  output_dir=$OUTPUT_DIR"
echo "  feature_cache_dir=$FEATURE_CACHE_DIR"
echo "  num_views=2"
nvidia-smi || true

python experiments/probe3d/scripts/check_scrream_adapter_training_inputs.py \
  --adapter_data "$ADAPTER_DATA" \
  --nova_ckpt "$NOVA_CKPT" \
  --num_queries "${SCRREAM_NUM_QUERIES:-10000}" \
  --split "${SCRREAM_TRAIN_SPLIT:-train}"

read -r -a TRAIN_EXTRA_ARGS <<< "${SCRREAM_TRAIN_EXTRA_ARGS:-}"
TRACKING_ARGS=()
if [[ "${SCRREAM_SWANLAB:-0}" != "0" ]]; then
  TRACKING_ARGS=(
    --swanlab
    --swanlab_project "${SCRREAM_SWANLAB_PROJECT:-PSUVPSC3DD}"
    --swanlab_experiment "${SCRREAM_SWANLAB_EXPERIMENT:-scrream_mesh_complete_smoke_mlp}"
  )
  if [[ -n "${SCRREAM_SWANLAB_WORKSPACE:-}" ]]; then
    TRACKING_ARGS+=(--swanlab_workspace "$SCRREAM_SWANLAB_WORKSPACE")
  fi
fi

python experiments/probe3d/train_vggt_nova_adapter.py \
  --dataset scrream_adapter \
  --data_root "$ADAPTER_DATA" \
  --nova_ckpt "$NOVA_CKPT" \
  --adapter_type mlp \
  --adapter_layers "${SCRREAM_ADAPTER_LAYERS:-4}" \
  --adapter_hidden_dim "${SCRREAM_ADAPTER_HIDDEN_DIM:-1024}" \
  --loss_type "${SCRREAM_LOSS_TYPE:-nova_flow}" \
  --num_queries "${SCRREAM_NUM_QUERIES:-10000}" \
  --num_views 2 \
  --train_split "${SCRREAM_TRAIN_SPLIT:-train}" \
  --val_split "${SCRREAM_VAL_SPLIT:-val}" \
  --test_split "${SCRREAM_TEST_SPLIT:-test}" \
  --batch_size "${SCRREAM_BATCH_SIZE:-1}" \
  --num_workers "${SCRREAM_NUM_WORKERS:-2}" \
  --lr "${SCRREAM_LR:-1e-4}" \
  --max_steps "${SCRREAM_MAX_STEPS:-20}" \
  --save_every "${SCRREAM_SAVE_EVERY:-20}" \
  --val_every "${SCRREAM_VAL_EVERY:-20}" \
  --eval_batches "${SCRREAM_EVAL_BATCHES:-1}" \
  --save_ply_queries "${SCRREAM_SAVE_PLY_QUERIES:-10000}" \
  --feature_cache_dir "$FEATURE_CACHE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${VGGT_ARGS[@]}" \
  "${TRACKING_ARGS[@]}" \
  "${TRAIN_EXTRA_ARGS[@]}"
