#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_flf2v_cache
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/feature_cache

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

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

ADAPTER_DATA="${WAN_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
MODEL_ID="${WAN_FLF2V_MODEL_ID:-checkpoints/wan2.1/Wan2.1-FLF2V-14B-720P-diffusers}"
CACHE_DIR="${WAN_FLF2V_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_flf2v14b_pair_endpoint81_480}"
TIMESTEPS="${WAN_FLF2V_TIMESTEPS:-249}"
LAYERS="${WAN_FLF2V_LAYERS:-14}"
HEIGHT="${WAN_HEIGHT:-480}"
WIDTH="${WAN_WIDTH:-832}"
MAX_SAMPLES="${WAN_MAX_SAMPLES:-0}"
SAMPLE_START="${WAN_SAMPLE_START:-0}"
SAMPLE_END="${WAN_SAMPLE_END:-0}"
DTYPE="${WAN_DTYPE:-bf16}"
DEVICE="${WAN_DEVICE:-cuda}"
SCHEDULER="${WAN_SCHEDULER:-checkpoint}"

EXTRA_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ "$SAMPLE_START" != "0" ]]; then
  EXTRA_ARGS+=(--sample_start "$SAMPLE_START")
fi
if [[ "$SAMPLE_END" != "0" ]]; then
  EXTRA_ARGS+=(--sample_end "$SAMPLE_END")
fi
if [[ "${WAN_WINDOW_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--window_only)
fi
if [[ "${WAN_FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi
if [[ "${WAN_CPU_OFFLOAD:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--cpu_offload)
fi

echo "SCRREAM WAN FLF2V feature precompute"
echo "  adapter_data=$ADAPTER_DATA"
echo "  cache_dir=$CACHE_DIR"
echo "  model_id=$MODEL_ID"
echo "  timesteps=$TIMESTEPS"
echo "  layers=$LAYERS"
echo "  resolution=${WIDTH}x${HEIGHT}"
echo "  sample_range=${SAMPLE_START}:${SAMPLE_END}"
echo "  max_samples=$MAX_SAMPLES"
echo "  dtype=$DTYPE"
echo "  scheduler=$SCHEDULER"
echo "  cpu_offload=${WAN_CPU_OFFLOAD:-1}"
echo "  proxy=${HTTP_PROXY:-<disabled>}"
echo "  proxy_tunnel_pid=${WAN_PROXY_TUNNEL_PID:-<none>}"
nvidia-smi || true

python -u experiments/probe3d/scripts/prepare_scrream_wan_flf2v_feature_cache.py \
  --adapter_data "$ADAPTER_DATA" \
  --output_dir "$CACHE_DIR" \
  --model_id "$MODEL_ID" \
  --timesteps "$TIMESTEPS" \
  --layers "$LAYERS" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --scheduler "$SCHEDULER" \
  "${EXTRA_ARGS[@]}"
