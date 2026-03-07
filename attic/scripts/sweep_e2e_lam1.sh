#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/e2e_lam1_%j.out
#SBATCH --error=logs/slurm/e2e_lam1_%j.err
#SBATCH --job-name=e2e_lam1

set -e
mkdir -p logs/slurm

module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes
export PYTHONUNBUFFERED=1

for PROBLEM in circle bimodal_quadratic; do
    echo "=========================================="
    echo "Problem: $PROBLEM | Flow lam=1.0, L=10 | GPA L=1000, eta=0.005"
    echo "=========================================="

    OUT_DIR="results/${PROBLEM}/gpa_plots/13_e2e_lam1_L1000"
    mkdir -p "$OUT_DIR"

    python -u run_end_to_end.py \
        --problem "$PROBLEM" \
        --flow-iters 20000 \
        --flow-lam 1.0 \
        --flow-lip-scale 10.0 \
        --flow-gp-lambda 1.0 \
        --flow-n-steps 10 \
        --K 500 \
        --eta 0.005 \
        --disc-steps 10 \
        --L 1000 \
        --n-train 10000 \
        --n-eval 2000 \
        --output-dir "$OUT_DIR" \
        --seed 42

    echo "Done: $PROBLEM"
    echo ""
done

echo "All done!"
