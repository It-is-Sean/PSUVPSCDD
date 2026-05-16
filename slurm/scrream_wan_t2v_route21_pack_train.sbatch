#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_r21_pack
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

if [[ "${WAN_DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
else
  WAN_PROXY_REMOTE_PORT="${WAN_PROXY_REMOTE_PORT:-17890}"
  WAN_PROXY_LOCAL_PORT="${WAN_PROXY_LOCAL_PORT:-17890}"
  WAN_PROXY_SSH_HOST="${WAN_PROXY_SSH_HOST:-air-server}"
  WAN_PROXY_TUNNEL_PID=""
  if [[ "${WAN_PROXY_SSH_TUNNEL:-1}" == "1" && "$(hostname -s)" != "$WAN_PROXY_SSH_HOST" ]]; then
    ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -N \
      -L "127.0.0.1:${WAN_PROXY_LOCAL_PORT}:127.0.0.1:${WAN_PROXY_REMOTE_PORT}" \
      "$WAN_PROXY_SSH_HOST" &
    WAN_PROXY_TUNNEL_PID="$!"
    trap 'if [[ -n "${WAN_PROXY_TUNNEL_PID:-}" ]]; then kill "$WAN_PROXY_TUNNEL_PID" 2>/dev/null || true; fi' EXIT
    sleep 2
  fi
  export HTTP_PROXY="http://127.0.0.1:${WAN_PROXY_LOCAL_PORT}"
  export HTTPS_PROXY="$HTTP_PROXY"
  export ALL_PROXY="$HTTP_PROXY"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  export all_proxy="$ALL_PROXY"
fi

NOISE_MODE="${WAN_NOISE_MODE:?Set WAN_NOISE_MODE to no_noise or low_noise for Route2.1}"
if [[ "$NOISE_MODE" != "no_noise" && "$NOISE_MODE" != "low_noise" ]]; then
  echo "Route2.1 pack only supports WAN_NOISE_MODE=no_noise or low_noise, got $NOISE_MODE" >&2
  exit 2
fi

TIMESTEP="${WAN_TIMESTEP:-999}"
LAYERS_CSV="${WAN_LAYERS:-9,14,29}"
IFS=',' read -r -a LAYERS <<< "$LAYERS_CSV"

VISIBLE_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$VISIBLE_RAW" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$VISIBLE_RAW"
else
  GPU_COUNT="${SLURM_GPUS_ON_NODE:-${WAN_PACK_GPUS:-1}}"
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
  done
fi
MAX_PARALLEL="${WAN_PACK_PARALLEL:-${#GPU_IDS[@]}}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "No GPUs visible for WAN Route2.1 packed ablation." >&2
  exit 2
fi

ADAPTER_DATA="${WAN_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${WAN_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
if [[ "$NOISE_MODE" == "no_noise" ]]; then
  CACHE_DIR="${WAN_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81_no_noise}"
else
  CACHE_DIR="${WAN_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81_low_noise}"
fi
RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_seed17_wan_t2v1p3b_ctx81_${NOISE_MODE}_}"
NUM_QUERIES="${WAN_NUM_QUERIES:-20000}"
EPOCHS="${WAN_EPOCHS:-50}"
BATCH_SIZE="${WAN_BATCH_SIZE:-1}"
TRAIN_COUNT="${WAN_TRAIN_COUNT:-317}"
MAX_STEPS="${WAN_MAX_STEPS:-$(
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
  echo "Missing WAN Route2.1 feature cache dir: $CACHE_DIR" >&2
  exit 2
fi

echo "Packed SCRREAM WAN T2V Route2.1 ablation"
echo "  noise_mode=$NOISE_MODE"
echo "  timestep=$TIMESTEP"
echo "  layers=$LAYERS_CSV"
echo "  visible_gpus=${GPU_IDS[*]}"
echo "  max_parallel=$MAX_PARALLEL"
echo "  adapter_data=$ADAPTER_DATA"
echo "  cache_dir=$CACHE_DIR"
echo "  max_steps=$MAX_STEPS"
echo "  num_queries=$NUM_QUERIES"
echo "  eval_batches=${WAN_EVAL_BATCHES:-0}"
echo "  val_metric_max_points=${WAN_VAL_METRIC_MAX_POINTS:-20000}"
echo "  val_preview_samples=${WAN_VAL_PREVIEW_SAMPLES:-12}"
echo "  val_preview_queries=${WAN_VAL_PREVIEW_QUERIES:-40960}"
echo "  proxy=${HTTP_PROXY:-<disabled>}"
echo "  proxy_tunnel_pid=${WAN_PROXY_TUNNEL_PID:-<none>}"
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

job_idx=0
for layer in "${LAYERS[@]}"; do
  wait_for_slot
  layer_padded="$(printf '%02d' "$layer")"
  run_name="${RUN_PREFIX}layer${layer_padded}"
  gpu="${GPU_IDS[$((job_idx % ${#GPU_IDS[@]}))]}"
  output_dir="${WAN_OUTPUT_BASE:-experiments/probe3d/result}/${run_name}"
  mkdir -p "$output_dir"
  log_file="slurm_out/scrream_wan_t2v_route21_${NOISE_MODE}_layer${layer_padded}_${SLURM_JOB_ID:-manual}.log"
  echo "Launching noise_mode=$NOISE_MODE layer=$layer on CUDA_VISIBLE_DEVICES=$gpu output=$output_dir log=$log_file"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u experiments/probe3d/train_wan_t2v_nova_adapter.py \
      --dataset scrream_adapter \
      --data_root "$ADAPTER_DATA" \
      --nova_ckpt "$NOVA_CKPT" \
      --wan_feature_cache_dir "$CACHE_DIR" \
      --wan_timestep "$TIMESTEP" \
      --wan_layer "$layer" \
      --adapter_type mlp \
      --adapter_layers "${WAN_ADAPTER_LAYERS:-4}" \
      --adapter_hidden_dim "${WAN_ADAPTER_HIDDEN_DIM:-1024}" \
      --loss_type "${WAN_LOSS_TYPE:-nova_flow}" \
      --num_queries "$NUM_QUERIES" \
      --num_views 2 \
      --train_split "${WAN_TRAIN_SPLIT:-train}" \
      --val_split "${WAN_VAL_SPLIT:-val}" \
      --test_split "${WAN_TEST_SPLIT:-test}" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "${WAN_NUM_WORKERS:-4}" \
      --lr "${WAN_LR:-1e-4}" \
      --max_steps "$MAX_STEPS" \
      --save_every "${WAN_SAVE_EVERY:-500}" \
      --val_every "${WAN_VAL_EVERY:-500}" \
      --eval_batches "${WAN_EVAL_BATCHES:-0}" \
      --val_flow_t_bins "${WAN_VAL_FLOW_T_BINS:-10}" \
      --val_metric_max_points "${WAN_VAL_METRIC_MAX_POINTS:-20000}" \
      --val_preview_samples "${WAN_VAL_PREVIEW_SAMPLES:-12}" \
      --val_preview_queries "${WAN_VAL_PREVIEW_QUERIES:-40960}" \
      --save_ply_queries "${WAN_SAVE_PLY_QUERIES:-40960}" \
      --output_dir "$output_dir" \
      --swanlab \
      --swanlab_project "${WAN_SWANLAB_PROJECT:-PSUVPSC3DD}" \
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
