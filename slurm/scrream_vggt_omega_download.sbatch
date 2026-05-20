#!/usr/bin/env bash
#SBATCH --job-name=scrream_vggt_omega_download
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out checkpoints/vggt_omega

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

TARGET="${SCRREAM_VGGT_OMEGA_WEIGHTS:-checkpoints/vggt_omega/vggt_omega_1b_512.pt}"
REPO_ID="${SCRREAM_VGGT_OMEGA_REPO_ID:-facebook/VGGT-Omega}"
FILENAME="${SCRREAM_VGGT_OMEGA_FILENAME:-vggt_omega_1b_512.pt}"
export TARGET REPO_ID FILENAME

echo "Downloading VGGT-Omega checkpoint"
echo "  repo_id=$REPO_ID"
echo "  filename=$FILENAME"
echo "  target=$TARGET"

python - <<'PY'
from pathlib import Path
import os
import shutil

from huggingface_hub import hf_hub_download

target = Path(os.environ["TARGET"])
target.parent.mkdir(parents=True, exist_ok=True)
downloaded = Path(
    hf_hub_download(
        repo_id=os.environ["REPO_ID"],
        filename=os.environ["FILENAME"],
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
)
if downloaded.resolve() != target.resolve():
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(downloaded, tmp)
    tmp.replace(target)
print(target)
PY

python - <<'PY'
from pathlib import Path
import torch

target = Path(os.environ["TARGET"])
state = torch.load(target, map_location="cpu")
print(f"checkpoint={target} keys={len(state)}")
print("register_token", tuple(state["aggregator.register_token"].shape))
print("patch_proj", tuple(state["aggregator.patch_embed.patch_embed.proj.weight"].shape))
print("has_inter_frame", any(key.startswith("aggregator.inter_frame_blocks.") for key in state))
print("has_dense_head", any(key.startswith("dense_head.") for key in state))
PY
