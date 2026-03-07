#!/bin/bash
#SBATCH --job-name=bimodal_sweep
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

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: Conda environment not found at $CONDA_ENV"
    exit 1
fi
conda activate "$CONDA_ENV"

python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

export PYTHONUNBUFFERED=1

# Sweep: gp_lambda x lam x quad_features
N_ITERS=20000
BATCH_SIZE=256
N_STEPS=10

gp_lambdas=(1.0 5.0 10.0 50.0)
lams=(0.001 0.01 0.1)
lip_scales=(1.0 5.0 10.0)

gp_total=$(( ${#gp_lambdas[@]} * ${#lams[@]} ))
sn_total=$(( ${#lip_scales[@]} * ${#lams[@]} ))
total=$(( gp_total + sn_total ))
count=0

# --- GP sweep ---
echo "=== Gradient Penalty sweep: ${gp_total} runs ==="
for gp in "${gp_lambdas[@]}"; do
    for lam in "${lams[@]}"; do
        count=$((count + 1))

        echo ""
        echo "================================================================"
        echo "[$count/$total] GP: gp_lambda=$gp, lam=$lam"
        echo "================================================================"

        python -u run_experiment.py \
            --problem bimodal_quadratic \
            --n-iters $N_ITERS \
            --batch-size $BATCH_SIZE \
            --gp-lambda $gp \
            --lam $lam \
            --n-steps $N_STEPS \
            --vel-layers 4 \
            --vel-hidden 256 \
            --device cuda \
            --output-dir results/bimodal_quadratic \
            --checkpoint-every 5000

        echo "[$count/$total] Done."
    done
done

# --- Spectral norm sweep ---
echo ""
echo "=== Spectral Norm sweep: ${sn_total} runs ==="
for lip in "${lip_scales[@]}"; do
    for lam in "${lams[@]}"; do
        count=$((count + 1))

        echo ""
        echo "================================================================"
        echo "[$count/$total] SN: lip=$lip, lam=$lam"
        echo "================================================================"

        python -u run_experiment.py \
            --problem bimodal_quadratic \
            --n-iters $N_ITERS \
            --batch-size $BATCH_SIZE \
            --lip-scale $lip \
            --lam $lam \
            --n-steps $N_STEPS \
            --vel-layers 4 \
            --vel-hidden 256 \
            --device cuda \
            --output-dir results/bimodal_quadratic \
            --checkpoint-every 5000

        echo "[$count/$total] Done."
    done
done

echo ""
echo "=== All $total runs complete ==="
