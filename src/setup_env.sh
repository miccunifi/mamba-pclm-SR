#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mambair-sr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
GPU_PROFILE="${GPU_PROFILE:-auto}"
CLONE_REPOS=0
RUN_TEST=0
for a in "$@"; do
  case "$a" in
    --clone-repos) CLONE_REPOS=1 ;;
    --test) RUN_TEST=1 ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "argomento sconosciuto: $a" >&2; exit 1 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

log() { echo -e "\n[setup_env] $*"; }

# ---------------------------------------------------------------------------------------
# 0) conda
# ---------------------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda non trovato nel PATH" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

COMPUTE_CAP=""
if nvidia-smi >/dev/null 2>&1; then
  log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  log "compute capability: ${COMPUTE_CAP:-sconosciuta}"
else
  log "ATTENZIONE: nvidia-smi non disponibile; l'installazione procede ma i kernel Mamba richiedono una GPU"
fi

# Auto profile: compute capability >= 10 (Blackwell) needs the newer torch/CUDA stack.
if [ "$GPU_PROFILE" = "auto" ]; then
  major="${COMPUTE_CAP%%.*}"
  if [ -n "$major" ] && [ "$major" -ge 10 ] 2>/dev/null; then GPU_PROFILE=blackwell; else GPU_PROFILE=ampere; fi
fi
case "$GPU_PROFILE" in
  ampere)
    TORCH_VER=2.6.0; TV_VER=0.21.0; CU_TAG=cu124; MAMBA_VER=2.2.4; CC1D_VER=1.5.0.post5; NVCC_VER=12.4 ;;
  blackwell)
    TORCH_VER=2.8.0; TV_VER=0.23.0; CU_TAG=cu128; MAMBA_VER=2.2.5; CC1D_VER=1.5.2; NVCC_VER=12.8 ;;
  *) echo "GPU_PROFILE non valido: $GPU_PROFILE (ampere|blackwell)" >&2; exit 1 ;;
esac
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/${CU_TAG}}"
log "profilo: $GPU_PROFILE -> torch ${TORCH_VER}+${CU_TAG}, torchvision ${TV_VER}, mamba-ssm ${MAMBA_VER}, causal-conv1d ${CC1D_VER}"

# ---------------------------------------------------------------------------------------
# 1) env + python
# ---------------------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "env '$ENV_NAME' gia' esistente: lo riuso (pacchetti mancanti verranno aggiunti)"
else
  log "creo l'env '$ENV_NAME' (python $PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi
conda activate "$ENV_NAME"
PY="$(command -v python)"
log "python: $PY ($(python --version))"
python -m pip install --upgrade "pip<25" wheel setuptools packaging ninja

# ---------------------------------------------------------------------------------------
# 2) torch (cu124)
# ---------------------------------------------------------------------------------------
log "installo torch ${TORCH_VER} / torchvision ${TV_VER} (${CU_TAG})"
python -m pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" --index-url "$TORCH_INDEX"
python - <<'EOF'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | disponibile: {torch.cuda.is_available()} | "
      f"archs: {' '.join(torch.cuda.get_arch_list())}")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    ok = f"sm_{cc[0]}{cc[1]}" in torch.cuda.get_arch_list()
    print(f"GPU sm_{cc[0]}{cc[1]} supportata da questo torch: {ok}")
    if not ok:
        raise SystemExit("questo torch non ha kernel per la GPU: rilancia con GPU_PROFILE=blackwell (o ampere)")
EOF

export MAX_JOBS="${MAX_JOBS:-4}"
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"; export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

# $1 = FALSE tries the prebuilt wheels, TRUE forces a build of the CUDA kernels from source.
install_mamba() {
  export CAUSAL_CONV1D_FORCE_BUILD="$1" MAMBA_FORCE_BUILD="$1"
  python -m pip install --no-build-isolation --force-reinstall --no-deps "causal-conv1d==${CC1D_VER}"
  python -m pip install --no-build-isolation --force-reinstall --no-deps "mamba-ssm==${MAMBA_VER}"
}

# Returns 0 when the Mamba CUDA kernels run on the GPU (also 0 when no GPU is visible).
kernel_ok() {
  python - <<'EOF' 2>/dev/null
import torch, sys
if not torch.cuda.is_available():
    sys.exit(0)
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn
u = torch.randn(1, 22, 64, device='cuda'); d = torch.randn(1, 22, 64, device='cuda')
A = -torch.rand(22, 10, device='cuda'); B = torch.randn(1, 10, 64, device='cuda')
C = torch.randn(1, 10, 64, device='cuda'); D = torch.randn(22, device='cuda')
y = selective_scan_fn(u, d, A, B, C, D, delta_softplus=True)
z = causal_conv1d_fn(u, torch.randn(22, 4, device='cuda'), None)
torch.cuda.synchronize()
sys.exit(0 if torch.isfinite(y).all() and torch.isfinite(z).all() else 1)
EOF
}

# Try the prebuilt wheels first; compile from source only if the kernels fail on this GPU.
log "installo causal-conv1d ${CC1D_VER} e mamba-ssm ${MAMBA_VER} (wheel precompilato se disponibile)"
install_mamba FALSE || true
if kernel_ok; then
  log "kernel Mamba OK sulla GPU"
else
  log "il wheel non funziona su questa GPU (o non esiste): compilo dai sorgenti (10-30 min)"
  if ! command -v nvcc >/dev/null 2>&1; then
    log "installo nvcc ${NVCC_VER} nell'env (canale nvidia)"
    conda install -y -c nvidia "cuda-nvcc=${NVCC_VER}" "cuda-toolkit=${NVCC_VER}" "cuda-cudart-dev=${NVCC_VER}"
    export CUDA_HOME="$CONDA_PREFIX"
  fi
  # Build only for the detected architecture to keep compile time short.
  if [ -n "$COMPUTE_CAP" ]; then export TORCH_CUDA_ARCH_LIST="${COMPUTE_CAP}"; fi
  install_mamba TRUE
  kernel_ok || { echo "kernel Mamba ancora KO dopo la build: controlla nvcc/CUDA_HOME" >&2; exit 1; }
  log "kernel Mamba compilati e verificati"
fi

log "installo requirements.txt"
python -m pip install -r "$HERE/requirements.txt"
# requirements.txt must not have replaced the pinned torch build.
python -c "import torch, torchvision; assert torch.__version__.startswith('${TORCH_VER}'), torch.__version__"

if [ "$CLONE_REPOS" = "1" ]; then
  log "clono i repo delle baseline in $HERE/repos"
  mkdir -p "$HERE/repos"
  clone_at() {
    local d="$HERE/repos/$1"
    if [ -d "$d/.git" ]; then echo "  $1: presente"; return; fi
    git clone --quiet "$2" "$d"
    git -C "$d" checkout --quiet "$3" || echo "  ATTENZIONE: commit $3 non trovato in $1, resto su HEAD"
    echo "  $1 @ $(git -C "$d" rev-parse --short=12 HEAD)"
  }
  clone_at MambaIR            https://github.com/csguoh/MambaIR                 33d7b3460c4665229334cc8de38c7f4c766ed3be
  clone_at SAFMN              https://github.com/sunny2109/SAFMN                1a412066e927da640bc7e2a53ea60f9ffc134241
  clone_at DVMSR              https://github.com/nathan66666/DVMSR              86663ce906aab91c31ee5bf69bebc1ad61537405
  clone_at SwinIR             https://github.com/JingyunLiang/SwinIR            6545850fbf8df298df73d81f3e8cba638787c8bd
  clone_at IMDN               https://github.com/Zheng222/IMDN                  8f158e6a5ac9db6e5857d9159fd4a6c4214da574
  clone_at RLFN               https://github.com/bytedance/RLFN                 bccfde91bf8b2de80789ec033a029e6f2a932330
  clone_at CARN-pytorch       https://github.com/nmhkahn/CARN-pytorch           40d3514f81cccd680d846cd5ab07af35f2c8e0e2
  clone_at Simple-SR          https://github.com/Jia-Research-Lab/Simple-SR     08c71e9e46ba781df50893f0476ecd0fc004aa45
  clone_at LatticeNet-pytorch https://github.com/ymff0592/super-resolution      7bb1ce05566bc162b112596be30e381b163541f7
fi

log "verifica import e kernel CUDA della selective scan"
python - <<'EOF'
import importlib, sys
mods = ['torch', 'torchvision', 'mamba_ssm', 'causal_conv1d', 'basicsr', 'fvcore', 'thop', 'timm',
        'einops', 'torchsr', 'yaml', 'pandas', 'matplotlib', 'cv2', 'skimage', 'easydict', 'pytest']
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad.append(m); print(f"  {m:<14} ERRORE: {e}")
import torch
if torch.cuda.is_available():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    u = torch.randn(1, 22, 64, device='cuda'); d = torch.randn(1, 22, 64, device='cuda')
    A = -torch.rand(22, 10, device='cuda'); B = torch.randn(1, 10, 64, device='cuda')
    C = torch.randn(1, 10, 64, device='cuda'); D = torch.randn(22, device='cuda')
    y = selective_scan_fn(u, d, A, B, C, D, delta_softplus=True)
    print(f"  selective_scan_fn (kernel CUDA, dim=22): OK {tuple(y.shape)} su {torch.cuda.get_device_name(0)}")
else:
    print("  (nessuna GPU visibile: kernel non testato)")
sys.exit(1 if bad else 0)
EOF

if [ "$RUN_TEST" = "1" ]; then
  log "test del progetto"
  python scripts/make_configs.py >/dev/null
  python -m pytest tests/test_pclmb_rho.py -q
fi

log "fatto. Attiva con:  conda activate $ENV_NAME   (poi lavora da $HERE)"
