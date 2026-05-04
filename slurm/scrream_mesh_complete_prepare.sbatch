#!/usr/bin/env bash
#SBATCH --job-name=scrream_mesh_prep
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/adapter_data

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7896}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7896}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7896}"

read -r -a EXTRA_ARGS <<< "${SCRREAM_EXTRA_ARGS:-}"
OUTPUT_PATH="${SCRREAM_ADAPTER_OUT:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt}"

python experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py \
  --data_root "${SCRREAM_ROOT:-$HOME/datasets/SCRREAM}" \
  --pair_list "${SCRREAM_PAIR_LIST:-data/scrream/scrream_n2_list.json}" \
  --output_path "$OUTPUT_PATH" \
  --target_source mesh_complete \
  --target_points "${SCRREAM_TARGET_POINTS:-10000}" \
  --mesh_sample_points "${SCRREAM_MESH_SAMPLE_POINTS:-250000}" \
  --split_seed "${SCRREAM_SPLIT_SEED:-17}" \
  --fps_device "${SCRREAM_FPS_DEVICE:-cpu}" \
  "${EXTRA_ARGS[@]}"

python experiments/probe3d/scripts/check_scrream_adapter_training_inputs.py \
  --adapter_data "$OUTPUT_PATH" \
  --num_queries "${SCRREAM_TARGET_POINTS:-10000}" \
  --split train
