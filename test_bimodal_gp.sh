#!/bin/bash
#SBATCH --job-name=bimodal_gp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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

echo "=== Bimodal Quadratic with Gradient Penalty (5000 iters) ==="
python -u run_experiment.py \
    --problem bimodal_quadratic \
    --n-iters 5000 \
    --batch-size 256 \
    --gp-lambda 10.0 \
    --quad-features \
    --n-steps 10 \
    --vel-layers 4 \
    --vel-hidden 256 \
    --device cuda \
    --output-dir results/bimodal_gp

echo ""
echo "=== Done ==="
