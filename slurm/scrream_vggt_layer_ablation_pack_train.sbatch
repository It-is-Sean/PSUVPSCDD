#!/usr/bin/env bash
#SBATCH --job-name=scrream_vggt_layer_pack
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

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

LAYERS_CSV="${SCRREAM_ABLATION_LAYERS:-0,4,8,12,16,20,24}"
IFS=',' read -r -a LAYERS <<< "$LAYERS_CSV"
VISIBLE_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$VISIBLE_RAW" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$VISIBLE_RAW"
else
  GPU_COUNT="${SLURM_GPUS_ON_NODE:-${SCRREAM_PACK_GPUS:-1}}"
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
  done
fi
MAX_PARALLEL="${SCRREAM_PACK_PARALLEL:-${#GPU_IDS[@]}}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "No GPUs visible for packed layer ablation." >&2
  exit 2
fi

ADAPTER_DATA="${SCRREAM_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${SCRREAM_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
VGGT_WEIGHTS="${SCRREAM_VGGT_WEIGHTS:-checkpoints/vggt/model.pt}"
RUN_PREFIX="${SCRREAM_ABLATION_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17_vggt_layer}"
NUM_QUERIES="${SCRREAM_NUM_QUERIES:-20000}"
EPOCHS="${SCRREAM_EPOCHS:-30}"
BATCH_SIZE="${SCRREAM_BATCH_SIZE:-1}"
TRAIN_COUNT="${SCRREAM_TRAIN_COUNT:-317}"
MAX_STEPS="${SCRREAM_MAX_STEPS:-$(
  TRAIN_COUNT="$TRAIN_COUNT" EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" python - <<'PY'
import math
import os
print(max(1, math.ceil(float(os.environ["TRAIN_COUNT"]) * float(os.environ["EPOCHS"]) / max(1, int(os.environ["BATCH_SIZE"])))))
PY
)}"

if [[ ! -f "$ADAPTER_DATA" ]]; then
  echo "Missing SCRREAM adapter dataset: $ADAPTER_DATA" >&2
  exit 2
fi
if [[ ! -f "$NOVA_CKPT" ]]; then
  echo "Missing NOVA checkpoint: $NOVA_CKPT" >&2
  exit 2
fi
if [[ ! -f "$VGGT_WEIGHTS" ]]; then
  echo "Missing VGGT weights: $VGGT_WEIGHTS" >&2
  exit 2
fi

echo "Packed SCRREAM VGGT layer ablation"
echo "  layers=$LAYERS_CSV"
echo "  visible_gpus=${GPU_IDS[*]}"
echo "  max_parallel=$MAX_PARALLEL"
echo "  adapter_data=$ADAPTER_DATA"
echo "  max_steps=$MAX_STEPS"
echo "  num_queries=$NUM_QUERIES"
echo "  eval_batches=${SCRREAM_EVAL_BATCHES:-0}"
echo "  val_metric_max_points=${SCRREAM_VAL_METRIC_MAX_POINTS:-20000}"
echo "  val_preview_samples=${SCRREAM_VAL_PREVIEW_SAMPLES:-12}"
echo "  val_preview_queries=${SCRREAM_VAL_PREVIEW_QUERIES:-40960}"
nvidia-smi || true

declare -a PIDS=()

wait_for_slot() {
  while [[ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
    local still=()
    local pid
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still+=("$pid")
      else
        wait "$pid"
      fi
    done
    PIDS=("${still[@]}")
    if [[ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]]; then
      sleep 10
    fi
  done
}

for idx in "${!LAYERS[@]}"; do
  wait_for_slot
  layer="${LAYERS[$idx]}"
  layer_padded="$(printf '%02d' "$layer")"
  gpu="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  output_dir="${SCRREAM_OUTPUT_BASE:-experiments/probe3d/result}/${RUN_PREFIX}${layer_padded}"
  feature_cache_dir="${SCRREAM_FEATURE_CACHE_BASE:-experiments/probe3d/feature_cache}/${RUN_PREFIX}${layer_padded}"
  mkdir -p "$output_dir" "$feature_cache_dir"
  log_file="slurm_out/scrream_vggt_layer_${layer_padded}_${SLURM_JOB_ID:-manual}.log"
  echo "Launching layer=$layer on CUDA_VISIBLE_DEVICES=$gpu output=$output_dir log=$log_file"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export FEATURE_MEMORY_CACHE_MAX="${FEATURE_MEMORY_CACHE_MAX:-0}"
    python -u experiments/probe3d/train_vggt_nova_adapter.py \
      --dataset scrream_adapter \
      --data_root "$ADAPTER_DATA" \
      --nova_ckpt "$NOVA_CKPT" \
      --vggt_weights "$VGGT_WEIGHTS" \
      --adapter_type mlp \
      --adapter_layers "${SCRREAM_ADAPTER_LAYERS:-4}" \
      --adapter_hidden_dim "${SCRREAM_ADAPTER_HIDDEN_DIM:-1024}" \
      --loss_type "${SCRREAM_LOSS_TYPE:-nova_flow}" \
      --num_queries "$NUM_QUERIES" \
      --vggt_layer "$layer" \
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
      --feature_cache_dir "$feature_cache_dir" \
      --output_dir "$output_dir" \
      --swanlab \
      --swanlab_project "${SCRREAM_SWANLAB_PROJECT:-PSUVPSC3DD}" \
      --swanlab_experiment "${RUN_PREFIX}${layer_padded}"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
