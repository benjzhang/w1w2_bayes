#!/bin/bash
#SBATCH -J circle_gp
#SBATCH -p gpu
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/circle_gp_%j.out
#SBATCH -e /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/circle_gp_%j.err

PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"
CONDA_ENV="/work/bjzhang_umass_edu/.conda/envs/w1w2_bayes"

mkdir -p "$PROJECT_DIR/logs/slurm"
cd "$PROJECT_DIR"

module load conda/latest
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

python -u run_experiment.py \
    --problem circle \
    --n-iters 20000 \
    --batch-size 256 \
    --lam 0.01 \
    --gp-lambda 1.0 \
    --lip-scale 10.0 \
    --n-steps 10 \
    --output-dir results/circle \
    --checkpoint-every 5000
