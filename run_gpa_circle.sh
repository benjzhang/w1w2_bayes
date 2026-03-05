#!/bin/bash
#SBATCH -J gpa_circle
#SBATCH -p gpu
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/gpa_circle_%j.out
#SBATCH -e /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/gpa_circle_%j.err

PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"
CONDA_ENV="/work/bjzhang_umass_edu/.conda/envs/w1w2_bayes"

mkdir -p "$PROJECT_DIR/logs/slurm"
cd "$PROJECT_DIR"

module load conda/latest
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

python -u run_gpa_refine.py \
    --checkpoint results/circle/checkpoints/it20000_lam0.01_gp1.0_v3x128_st10/checkpoint_iter20000.pt \
    --problem circle \
    --no-warmstart \
    --K 500 \
    --eta 0.1 \
    --disc-steps 20 \
    --disc-lr 0.001 \
    --L 10.0 \
    --gp-weight 1.0 \
    --n-train 10000 \
    --batch-size 256 \
    --output-dir results/circle
