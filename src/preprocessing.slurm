#!/bin/bash
#SBATCH --job-name=iffar_preprocess
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try25_santini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_work/try25_santini/Deep-Learning/logs/preprocess_%j.out
#SBATCH --error=/leonardo_work/try25_santini/Deep-Learning/logs/preprocess_%j.err

# =========================
# 1. Setup ambiente
# =========================
set -euo pipefail
module purge
module load python/3.11.7

# Attiva l’ambiente virtuale corretto
source /leonardo_work/try25_santini/envs/audioenv/bin/activate

# Imposta le variabili per evitare oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# =========================
# 2. Directory di lavoro
# =========================
PROJECT_DIR="/leonardo_work/try25_santini/Deep-Learning"
cd "$PROJECT_DIR"

echo "[INFO] Working directory: $(pwd)"
echo "[INFO] Using Python from: $(which python)"
echo "[INFO] Starting preprocessing..."

# =========================
# 3. Esegui il preprocessing
# =========================
python -u src/preprocessing.py

echo "[INFO] ✅ Preprocessing completed."

