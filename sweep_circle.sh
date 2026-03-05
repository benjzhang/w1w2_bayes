#!/bin/bash
#SBATCH --job-name=circle_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/%x_%j.out
#SBATCH --error=/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/%x_%j.err

set -e

PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"
CONDA_ENV="/work/bjzhang_umass_edu/.conda/envs/w1w2_bayes"

mkdir -p "$PROJECT_DIR/logs/slurm"
cd "$PROJECT_DIR"

module load conda/latest
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

N_ITERS=20000
BATCH_SIZE=256
N_STEPS=10
OUTPUT_DIR="results/circle"

RUN=0

# --- GP sweep: gp_lambda x lam ---
GP_LAMBDAS=(1.0 5.0 10.0 50.0)
LAMS=(0.001 0.01 0.1)

for gp in "${GP_LAMBDAS[@]}"; do
    for lam in "${LAMS[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo "=========================================="
        echo "Run $RUN: GP lambda=$gp, lam=$lam"
        echo "=========================================="
        python -u run_experiment.py \
            --problem circle \
            --n-iters $N_ITERS \
            --batch-size $BATCH_SIZE \
            --lam $lam \
            --gp-lambda $gp \
            --n-steps $N_STEPS \
            --output-dir $OUTPUT_DIR \
            --checkpoint-every 5000
    done
done

echo ""
echo "=========================================="
echo "Sweep complete! Total runs: $RUN"
echo "=========================================="
