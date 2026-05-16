#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_t2v_cache
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

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
MODEL_ID="${WAN_MODEL_ID:-checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers}"
LAYERS="${WAN_LAYERS:-9,14,19,24,29}"
FEATURE_KIND="${WAN_FEATURE_KIND:-hidden}"
if [[ "$FEATURE_KIND" == "pred_x0_latent" ]]; then
  CACHE_DIR="${WAN_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81_pred_x0_latent}"
else
  CACHE_DIR="${WAN_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81}"
fi
NOISE_MODE="${WAN_NOISE_MODE:-normal}"
LOW_NOISE_INDEX="${WAN_LOW_NOISE_INDEX:-999}"
WINDOW_MODE="${WAN_WINDOW_MODE:-ctx81}"
if [[ "$NOISE_MODE" == "normal" ]]; then
  TIMESTEPS="${WAN_TIMESTEPS:-249,499,749}"
else
  TIMESTEPS="${WAN_TIMESTEPS:-$LOW_NOISE_INDEX}"
fi
MAX_SAMPLES="${WAN_MAX_SAMPLES:-0}"
DTYPE="${WAN_DTYPE:-bf16}"
DEVICE="${WAN_DEVICE:-cuda}"
VAE_DEVICE="${WAN_VAE_DEVICE:-cuda}"

EXTRA_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ "${WAN_WINDOW_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--window_only)
fi
if [[ "${WAN_FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

echo "SCRREAM WAN T2V feature precompute"
echo "  adapter_data=$ADAPTER_DATA"
echo "  cache_dir=$CACHE_DIR"
echo "  model_id=$MODEL_ID"
echo "  timesteps=$TIMESTEPS"
echo "  layers=$LAYERS"
echo "  feature_kind=$FEATURE_KIND"
echo "  noise_mode=$NOISE_MODE"
echo "  low_noise_index=$LOW_NOISE_INDEX"
echo "  window_mode=$WINDOW_MODE"
echo "  max_samples=$MAX_SAMPLES"
echo "  dtype=$DTYPE"
echo "  vae_device=$VAE_DEVICE"
echo "  proxy=${HTTP_PROXY:-<disabled>}"
echo "  proxy_tunnel_pid=${WAN_PROXY_TUNNEL_PID:-<none>}"
nvidia-smi || true

python -u experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py \
  --adapter_data "$ADAPTER_DATA" \
  --output_dir "$CACHE_DIR" \
  --model_id "$MODEL_ID" \
  --timesteps "$TIMESTEPS" \
  --layers "$LAYERS" \
  --feature_kind "$FEATURE_KIND" \
  --noise_mode "$NOISE_MODE" \
  --low_noise_index "$LOW_NOISE_INDEX" \
  --window_mode "$WINDOW_MODE" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --vae_device "$VAE_DEVICE" \
  "${EXTRA_ARGS[@]}"
