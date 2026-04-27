#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-nova3r}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ENV_FILE="${REPO_ROOT}/environment.yml"
USE_PROXY="${USE_PROXY:-auto}"
DEFAULT_PROXY="http://127.0.0.1:7890"
SYSTEM_CONDA_CANDIDATES=(
  "/data1/jcd_data/miniconda3/bin/conda"
  "/opt/conda/bin/conda"
  "/usr/local/bin/conda"
  "/usr/bin/conda"
  "$HOME/miniconda3/bin/conda"
  "$HOME/anaconda3/bin/conda"
)

log() { printf '[probe-env] %s\n' "$*"; }
warn() { printf '[probe-env][warn] %s\n' "$*" >&2; }

maybe_enable_proxy() {
  if [[ "${USE_PROXY}" == "0" || "${USE_PROXY}" == "false" || "${USE_PROXY}" == "off" ]]; then
    return 0
  fi
  if [[ -n "${HTTP_PROXY:-}" || -n "${HTTPS_PROXY:-}" || -n "${ALL_PROXY:-}" ]]; then
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 2 http://127.0.0.1:7890 >/dev/null 2>&1; then
      export HTTP_PROXY="${DEFAULT_PROXY}"
      export HTTPS_PROXY="${DEFAULT_PROXY}"
      export ALL_PROXY="${DEFAULT_PROXY}"
      export NO_PROXY="127.0.0.1,localhost"
      log "Enabled local proxy ${DEFAULT_PROXY}"
    fi
  fi
}

activate_conda() {
  local conda_bin
  if command -v conda >/dev/null 2>&1; then
    conda_bin="$(command -v conda)"
  else
    for candidate in "${SYSTEM_CONDA_CANDIDATES[@]}"; do
      if [[ -x "${candidate}" ]]; then
        conda_bin="${candidate}"
        break
      fi
    done
  fi

  if [[ ! -x "${conda_bin}" ]]; then
    echo "conda not found. Expected one of: ${SYSTEM_CONDA_CANDIDATES[*]}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  eval "$(${conda_bin} shell.bash hook)"
  export PATH="$(dirname "${conda_bin}"):${PATH}"
  log "Activated conda shell hook via ${conda_bin}"
}

ensure_env_file() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing environment file: ${ENV_FILE}" >&2
    exit 1
  fi
}

create_or_update_env() {
  ensure_env_file
  maybe_enable_proxy
  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    log "Updating existing conda env ${ENV_NAME} from environment.yml"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  else
    log "Creating conda env ${ENV_NAME} from environment.yml"
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
  fi
  conda activate "${ENV_NAME}"
  log "Conda env active: ${CONDA_DEFAULT_ENV}"
}

install_torch_cluster() {
  conda activate "${ENV_NAME}"
  if python -c 'import torch_cluster' >/dev/null 2>&1; then
    log "torch-cluster already installed"
    return 0
  fi
  local torch_short
  torch_short="$(python - <<'PY'
import torch
v=torch.__version__.split('+')[0].rsplit('.',1)[0]
print(v)
PY
)"
  log "Installing torch-cluster for torch ${torch_short}"
  pip install torch-cluster -f "https://data.pyg.org/whl/torch-${torch_short}.0+cu121.html" || pip install torch-cluster
}

install_pytorch3d_best_effort() {
  conda activate "${ENV_NAME}"
  if python -c 'import pytorch3d' >/dev/null 2>&1; then
    log "pytorch3d already installed"
    return 0
  fi
  log "Installing pytorch3d (best-effort)"
  if conda install -n "${ENV_NAME}" -y -c nvidia cuda-nvcc=12.1; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
  else
    warn "Failed to install cuda-nvcc=12.1 into env; will try build with current toolchain"
  fi
  FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-4}" pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" || \
    warn "pytorch3d build failed; current workflow can still fall back to matplotlib visualization"
}

compile_croco_best_effort() {
  conda activate "${ENV_NAME}"
  local rope_dir="${REPO_ROOT}/croco/models/curope"
  if [[ ! -d "${rope_dir}" ]]; then
    warn "CroCo RoPE directory missing: ${rope_dir}"
    return 0
  fi
  log "Compiling CroCo RoPE kernels (best-effort)"
  pushd "${rope_dir}" >/dev/null
  python setup.py build_ext --inplace || warn "RoPE CUDA kernels failed to build; inference may be slower"
  popd >/dev/null
}

install_chamferdist_best_effort() {
  conda activate "${ENV_NAME}"
  if python -c 'import chamferdist' >/dev/null 2>&1; then
    log "chamferdist already installed"
    return 0
  fi
  mkdir -p "${REPO_ROOT}/third_party"
  pushd "${REPO_ROOT}/third_party" >/dev/null
  if [[ ! -d chamferdist_custom ]]; then
    git clone https://github.com/wrchen530/chamferdist_custom.git
  fi
  pushd chamferdist_custom >/dev/null
  log "Installing chamferdist (best-effort)"
  python setup.py install || warn "chamferdist build failed; eval workflow may be incomplete"
  popd >/dev/null
  popd >/dev/null
}

verify_env() {
  conda activate "${ENV_NAME}"
  python "${REPO_ROOT}/scripts/probe/verify_env.py"
}

main() {
  activate_conda
  create_or_update_env
  install_torch_cluster
  install_pytorch3d_best_effort
  compile_croco_best_effort
  install_chamferdist_best_effort
  verify_env
  log "Done. Use: conda activate ${ENV_NAME}"
}

main "$@"
