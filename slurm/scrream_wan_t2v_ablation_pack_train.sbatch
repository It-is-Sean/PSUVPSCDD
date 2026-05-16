#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_t2v_pack
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

FEATURE_KIND="${WAN_FEATURE_KIND:-hidden}"
ADAPTER_TYPE="${WAN_ADAPTER_TYPE:-mlp}"
TIMESTEPS_CSV="${WAN_TIMESTEPS:-249,499,749}"
if [[ "$FEATURE_KIND" == "hidden" ]]; then
  LAYERS_CSV="${WAN_LAYERS:-9,14,19,24,29}"
else
  if [[ -n "${WAN_LAYERS:-}" && "${WAN_LAYERS}" != "0" ]]; then
    echo "Ignoring WAN_LAYERS=$WAN_LAYERS for feature_kind=$FEATURE_KIND; non-hidden WAN features have no block layer."
  fi
  LAYERS_CSV="0"
fi
IFS=',' read -r -a TIMESTEPS <<< "$TIMESTEPS_CSV"
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
  echo "No GPUs visible for WAN packed ablation." >&2
  exit 2
fi

ADAPTER_DATA="${WAN_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${WAN_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
if [[ "$FEATURE_KIND" == "pred_x0_latent" ]]; then
  CACHE_DIR="${WAN_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81_pred_x0_latent}"
  if [[ "$ADAPTER_TYPE" == "conv2d_mlp" ]]; then
    RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_pred_x0_latent_conv2d_}"
  else
    RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_pred_x0_latent_}"
  fi
else
  CACHE_DIR="${WAN_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81}"
  if [[ "$ADAPTER_TYPE" == "grid2d_pool" ]]; then
    RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_grid2d_}"
  elif [[ "$ADAPTER_TYPE" == "grid2d_conv" ]]; then
    RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_grid2d_conv_}"
  elif [[ "$ADAPTER_TYPE" == "wan_cross_attn_resampler" ]]; then
    RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_ca_resampler_}"
  else
    RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_seed17_wan_t2v1p3b_ctx81_}"
  fi
fi
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
  echo "Missing WAN feature cache dir: $CACHE_DIR" >&2
  exit 2
fi

echo "Packed SCRREAM WAN T2V route2 ablation"
echo "  timesteps=$TIMESTEPS_CSV"
echo "  layers=$LAYERS_CSV"
echo "  visible_gpus=${GPU_IDS[*]}"
echo "  max_parallel=$MAX_PARALLEL"
echo "  adapter_data=$ADAPTER_DATA"
echo "  cache_dir=$CACHE_DIR"
echo "  feature_kind=$FEATURE_KIND"
echo "  adapter_type=$ADAPTER_TYPE"
echo "  adapter_layers=${WAN_ADAPTER_LAYERS:-4}"
echo "  adapter_hidden_dim=${WAN_ADAPTER_HIDDEN_DIM:-1024}"
echo "  adapter_heads=${WAN_ADAPTER_HEADS:-8}"
echo "  adapter_mlp_ratio=${WAN_ADAPTER_MLP_RATIO:-2.0}"
echo "  max_steps=$MAX_STEPS"
echo "  num_queries=$NUM_QUERIES"
echo "  feature_norm=${WAN_FEATURE_NORM:-none}"
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
for timestep in "${TIMESTEPS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    wait_for_slot
    layer_padded="$(printf '%02d' "$layer")"
    if [[ "$FEATURE_KIND" == "hidden" ]]; then
      run_name="${RUN_PREFIX}t${timestep}_layer${layer_padded}"
    else
      run_name="${RUN_PREFIX}t${timestep}"
    fi
    gpu="${GPU_IDS[$((job_idx % ${#GPU_IDS[@]}))]}"
    output_dir="${WAN_OUTPUT_BASE:-experiments/probe3d/result}/${run_name}"
    mkdir -p "$output_dir"
    if [[ "$FEATURE_KIND" == "hidden" ]]; then
      log_file="slurm_out/scrream_wan_t2v_t${timestep}_layer${layer_padded}_${SLURM_JOB_ID:-manual}.log"
      echo "Launching t=$timestep layer=$layer on CUDA_VISIBLE_DEVICES=$gpu output=$output_dir log=$log_file"
    else
      log_file="slurm_out/scrream_wan_t2v_${FEATURE_KIND}_t${timestep}_${SLURM_JOB_ID:-manual}.log"
      echo "Launching t=$timestep feature_kind=$FEATURE_KIND on CUDA_VISIBLE_DEVICES=$gpu output=$output_dir log=$log_file"
    fi
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python -u experiments/probe3d/train_wan_t2v_nova_adapter.py \
        --dataset scrream_adapter \
        --data_root "$ADAPTER_DATA" \
        --nova_ckpt "$NOVA_CKPT" \
        --wan_feature_cache_dir "$CACHE_DIR" \
        --wan_timestep "$timestep" \
        --wan_layer "$layer" \
        --wan_feature_kind "$FEATURE_KIND" \
        --wan_feature_norm "${WAN_FEATURE_NORM:-none}" \
        --wan_feature_norm_eps "${WAN_FEATURE_NORM_EPS:-1e-6}" \
        --adapter_type "$ADAPTER_TYPE" \
        --adapter_layers "${WAN_ADAPTER_LAYERS:-4}" \
        --adapter_hidden_dim "${WAN_ADAPTER_HIDDEN_DIM:-1024}" \
        --adapter_heads "${WAN_ADAPTER_HEADS:-8}" \
        --adapter_mlp_ratio "${WAN_ADAPTER_MLP_RATIO:-2.0}" \
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
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
