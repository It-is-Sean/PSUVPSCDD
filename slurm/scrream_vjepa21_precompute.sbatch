#!/usr/bin/env bash
#SBATCH --job-name=scrream_vjepa21_precompute
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/Users/sean/Code/PSUVPSCDD}"
mkdir -p slurm_out experiments/probe3d/feature_cache

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate "${VJEPA21_CONDA_ENV:-nova3r}"

if [[ "${SCRREAM_DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
else
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7896}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7896}"
  export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7896}"
fi

WINDOW_MODES_CSV="${VJEPA21_WINDOW_MODES:-pair_exact16,ctx_anchor16,ctx_shuffle16,ctx_anchor32}"
IFS=',' read -r -a WINDOW_MODES <<< "$WINDOW_MODES_CSV"

for mode in "${WINDOW_MODES[@]}"; do
  echo "Precomputing V-JEPA 2.1 mode=$mode"
  cmd=(
    python -u experiments/probe3d/scripts/prepare_scrream_vjepa21_feature_cache.py
    --adapter_data "${VJEPA21_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
    --output_dir "${VJEPA21_CACHE_ROOT:-experiments/probe3d/feature_cache/scrream_vjepa21}"
    --checkpoint_path "${VJEPA21_CHECKPOINT:-checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt}"
    --model_name "${VJEPA21_MODEL_NAME:-vjepa2_1_vit_base_384}"
    --window_mode "$mode"
    --max_samples "${VJEPA21_MAX_SAMPLES:-0}"
    --device "${VJEPA21_DEVICE:-cuda}"
    --seed "${VJEPA21_SEED:-17}"
    --crop_size "${VJEPA21_CROP_SIZE:-384}"
    --patch_size "${VJEPA21_PATCH_SIZE:-16}"
    --tubelet_size "${VJEPA21_TUBELET_SIZE:-2}"
  )
  if [[ -n "${VJEPA21_SPLIT:-}" ]]; then
    cmd+=(--split "${VJEPA21_SPLIT}")
  fi
  "${cmd[@]}"
done
