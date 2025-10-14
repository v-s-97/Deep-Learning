#!/bin/bash
#SBATCH --job-name=iffar_ddp
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/try25_santini/Deep-Learning/logs/%x_%j.out
#SBATCH --error=/leonardo_work/try25_santini/Deep-Learning/logs/%x_%j.err
#SBATCH --chdir=/leonardo_work/try25_santini/Deep-Learning

# ================================================================
#  IFFAR — Distributed PyTorch Training on Leonardo (4×GPU)
# ================================================================

set -euo pipefail
echo "[slurm] Host: $(hostname)"
echo "[slurm] CWD:  $(pwd)"
echo "[slurm] JobID: ${SLURM_JOB_ID}"

# ------------------------
#  Environment setup
# ------------------------
module purge
module load cuda/12.3
module load python/3.11.7

VENV=/leonardo/home/userexternal/vsantini/iffar_env
source "$VENV/bin/activate"

echo "[env] Python: $(which python)"
echo "[env] Torchrun: $(which torchrun)"
echo "[env] GPU count: $SLURM_GPUS_ON_NODE"

# ------------------------
#  NCCL / PyTorch environment
# ------------------------
export OMP_NUM_THREADS=8
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISABLE_ADDR2LINE=1

# (Facoltativo ma consigliato per stabilità TCP)
export NCCL_IB_DISABLE=1

# ------------------------
#  Network autodetect
# ------------------------
echo "[net] Elenco interfacce IPv4:"
ip -br -4 addr || true
echo "[net] Default route:"
ip -o -4 route show to default || true

NET_IFACE=$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -n1)
if [[ -n "${NET_IFACE:-}" ]]; then
  echo "[net] Default network interface detected: $NET_IFACE"
  export NCCL_SOCKET_IFNAME="$NET_IFACE"
  export GLOO_SOCKET_IFNAME="$NET_IFACE"
else
  echo "[net] No default interface detected; letting NCCL autodetect"
  unset NCCL_SOCKET_IFNAME
  unset GLOO_SOCKET_IFNAME
fi

# ------------------------
#  Temporary directories
# ------------------------
export TMPDIR=$CINECA_SCRATCH/tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TORCHELASTIC_ERROR_FILE=/leonardo_work/try25_santini/Deep-Learning/logs/elastic_err_${SLURM_JOB_ID}.log
echo "[slurm] TMPDIR=$TMPDIR"

# ------------------------
#  Pre-run checks
# ------------------------
python - <<'EOF'
import torch, pathlib
print(f"[check] torch: {torch.__version__}")
print(f"[check] cuda available: {torch.cuda.is_available()}")
print(f"[check] visible devices: {torch.cuda.device_count()}")
for p in ["data", "manifests/sr16000", "src/train.py"]:
    print(f"[check] {p}: {'OK' if pathlib.Path(p).exists() else 'MISSING'}")
EOF

# ------------------------
#  Distributed launch
# ------------------------
echo "[slurm] Launching torchrun..."

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
echo "[slurm] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[slurm] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}  GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<unset>}"

torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --tee 3 \
  src/train.py

echo "[slurm] Job completed successfully."

