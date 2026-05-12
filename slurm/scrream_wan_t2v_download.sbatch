#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_t2v_download
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out checkpoints/wan2.1

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

MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
LOCAL_DIR="${WAN_LOCAL_DIR:-checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers}"
HF_MAX_WORKERS="${WAN_HF_MAX_WORKERS:-2}"

# The default hf-xet path can stall on this cluster through the SSH-forwarded
# proxy. Keep the download on the regular HuggingFace HTTP path and reduce
# concurrency so large shards make steady progress instead of holding many
# stale locks.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

echo "WAN T2V download/preflight"
echo "  model_id=$MODEL_ID"
echo "  local_dir=$LOCAL_DIR"
echo "  proxy=${HTTP_PROXY:-<disabled>}"
echo "  proxy_tunnel_pid=${WAN_PROXY_TUNNEL_PID:-<none>}"
echo "  hf_max_workers=$HF_MAX_WORKERS"
echo "  hf_hub_disable_xet=$HF_HUB_DISABLE_XET"

if [[ "${WAN_INSTALL_DEPS:-1}" == "1" ]]; then
  python -m pip install -r experiments/probe3d/requirements-wan-t2v.txt
fi

python - <<'PY'
import diffusers
import transformers
import tokenizers
from diffusers import WanPipeline
from transformers import UMT5EncoderModel
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("WanPipeline", WanPipeline)
print("UMT5EncoderModel", UMT5EncoderModel)
if WanPipeline.__module__.startswith("diffusers.utils.dummy"):
    raise SystemExit("Installed diffusers exposes only dummy WanPipeline; check transformers/torch compatibility")
PY

hf download "$MODEL_ID" \
  --local-dir "$LOCAL_DIR" \
  --max-workers "$HF_MAX_WORKERS" \
  --include \
    "model_index.json" \
    "scheduler/*" \
    "tokenizer/*" \
    "text_encoder/*" \
    "transformer/*" \
    "vae/*"

test -d "$LOCAL_DIR/transformer"
test -d "$LOCAL_DIR/vae"
test -d "$LOCAL_DIR/text_encoder"
test -f "$LOCAL_DIR/model_index.json"

python - <<PY
from pathlib import Path
root = Path("$LOCAL_DIR")
print("downloaded", root)
for path in ["model_index.json", "transformer", "vae", "text_encoder"]:
    p = root / path
    print(path, "ok" if p.exists() else "missing")
PY
