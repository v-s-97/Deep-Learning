#!/bin/bash
#SBATCH -J phase-eval-test
#SBATCH -A try25_santini
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:30:00
#SBATCH --mem=0
#SBATCH -o /leonardo_work/try25_santini/Deep-Learning/logs/%x-%j.out
#SBATCH -e /leonardo_work/try25_santini/Deep-Learning/logs/%x-%j.err

# --- Ambiente software ---
module purge
source /leonardo_work/try25_santini/envs/audioenv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --- Percorsi ---
cd /leonardo_work/try25_santini/Deep-Learning/src

# Tutto su WORK
OUTDIR="/leonardo_work/try25_santini/Deep-Learning/eval_out_${SLURM_JOB_ID}"
LOGDIR="/leonardo_work/try25_santini/Deep-Learning/logs"

mkdir -p "$OUTDIR" "$LOGDIR"

echo "Output directory: $OUTDIR"
echo "Logs directory:   $LOGDIR"
echo "Running on node:  $(hostname)"

# --- Lancio ---
srun python -u eval.py

