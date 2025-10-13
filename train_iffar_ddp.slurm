#!/bin/bash
#SBATCH --job-name=iffar_ddp
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load cuda/12.4
module load anaconda3
source activate /path/to/iffar_env

export OMP_NUM_THREADS=10
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

export TMPDIR=${CINECA_SCRATCH}/tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" logs

cd "$SLURM_SUBMIT_DIR"

torchrun --standalone --nproc_per_node=4 train.py
