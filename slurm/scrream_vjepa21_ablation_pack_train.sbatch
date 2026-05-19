#!/usr/bin/env bash
#SBATCH --job-name=scrream_vjepa21_pack
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/Users/sean/Code/PSUVPSCDD}"
mkdir -p slurm_out experiments/probe3d/result experiments/probe3d/feature_cache

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate "${VJEPA21_CONDA_ENV:-nova3r}"

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

WINDOW_MODES_CSV="${VJEPA21_WINDOW_MODES:-pair_exact16,ctx_anchor16,ctx_shuffle16,ctx_anchor32}"
IFS=',' read -r -a WINDOW_MODES <<< "$WINDOW_MODES_CSV"

VISIBLE_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$VISIBLE_RAW" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$VISIBLE_RAW"
else
  GPU_COUNT="${SLURM_GPUS_ON_NODE:-${VJEPA21_PACK_GPUS:-1}}"
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
  done
fi
MAX_PARALLEL="${VJEPA21_PACK_PARALLEL:-${#GPU_IDS[@]}}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "No GPUs visible for packed V-JEPA 2.1 ablation." >&2
  exit 2
fi

ADAPTER_DATA="${VJEPA21_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${VJEPA21_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
CACHE_DIR="${VJEPA21_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_vjepa21}"
RUN_PREFIX="${VJEPA21_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_vjepa21_}"
NUM_QUERIES="${VJEPA21_NUM_QUERIES:-20000}"
EPOCHS="${VJEPA21_EPOCHS:-50}"
BATCH_SIZE="${VJEPA21_BATCH_SIZE:-1}"
TRAIN_COUNT="${VJEPA21_TRAIN_COUNT:-317}"
MAX_STEPS="${VJEPA21_MAX_STEPS:-$(
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
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "Missing V-JEPA 2.1 feature cache dir: $CACHE_DIR" >&2
  exit 2
fi

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

job_idx=0
for mode in "${WINDOW_MODES[@]}"; do
  wait_for_slot
  gpu="${GPU_IDS[$((job_idx % ${#GPU_IDS[@]}))]}"
  run_name="${RUN_PREFIX}${mode}"
  output_dir="${VJEPA21_OUTPUT_BASE:-experiments/probe3d/result}/${run_name}"
  mkdir -p "$output_dir"
  log_file="slurm_out/scrream_vjepa21_${mode}_${SLURM_JOB_ID:-manual}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u experiments/probe3d/train_vjepa21_nova_adapter.py \
      --dataset scrream_adapter \
      --data_root "$ADAPTER_DATA" \
      --nova_ckpt "$NOVA_CKPT" \
      --vjepa_feature_cache_dir "$CACHE_DIR" \
      --vjepa_window_mode "$mode" \
      --vjepa_model_name "${VJEPA21_MODEL_NAME:-vjepa2_1_vit_large_384}" \
      --adapter_layers "${VJEPA21_ADAPTER_LAYERS:-4}" \
      --adapter_hidden_dim "${VJEPA21_ADAPTER_HIDDEN_DIM:-1024}" \
      --loss_type "${VJEPA21_LOSS_TYPE:-nova_flow}" \
      --num_queries "$NUM_QUERIES" \
      --num_views 2 \
      --train_split "${VJEPA21_TRAIN_SPLIT:-train}" \
      --val_split "${VJEPA21_VAL_SPLIT:-val}" \
      --test_split "${VJEPA21_TEST_SPLIT:-test}" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "${VJEPA21_NUM_WORKERS:-4}" \
      --lr "${VJEPA21_LR:-1e-4}" \
      --max_steps "$MAX_STEPS" \
      --save_every "${VJEPA21_SAVE_EVERY:-500}" \
      --val_every "${VJEPA21_VAL_EVERY:-500}" \
      --eval_batches "${VJEPA21_EVAL_BATCHES:-0}" \
      --val_flow_t_bins "${VJEPA21_VAL_FLOW_T_BINS:-10}" \
      --val_metric_max_points "${VJEPA21_VAL_METRIC_MAX_POINTS:-20000}" \
      --val_preview_samples "${VJEPA21_VAL_PREVIEW_SAMPLES:-12}" \
      --val_preview_queries "${VJEPA21_VAL_PREVIEW_QUERIES:-40960}" \
      --save_ply_queries "${VJEPA21_SAVE_PLY_QUERIES:-40960}" \
      --output_dir "$output_dir" \
      --swanlab \
      --swanlab_project "${VJEPA21_SWANLAB_PROJECT:-PSUVPSC3DD}" \
      --swanlab_experiment "$run_name"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  job_idx=$((job_idx + 1))
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
