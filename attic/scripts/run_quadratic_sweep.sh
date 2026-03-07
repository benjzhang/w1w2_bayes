#!/bin/bash
#
# Quadratic problem: sweep over λ and Lip scale
# Finds best hyperparameters for the parabola constraint
#
# Usage: sbatch run_quadratic_sweep.sh
#
#SBATCH --job-name=quad_sweep
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

# --- Setup ---
mkdir -p "$PROJECT_DIR/logs/slurm"
cd "$PROJECT_DIR"

module load conda/latest

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: Conda environment not found at $CONDA_ENV"
    echo "Run: bash attic/setup_env.sh"
    exit 1
fi
conda activate "$CONDA_ENV"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# --- Hyperparameter grid ---
lam_values=("0.001" "0.005" "0.01" "0.05")
lip_values=("5.0" "10.0" "20.0")
epochs=300

output_dir="experiments/quadratic_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output_dir/logs"

echo "Quadratic problem hyperparameter sweep"
echo "Output: $output_dir"
echo ""

for lam in "${lam_values[@]}"; do
    for lip in "${lip_values[@]}"; do
        log_file="$output_dir/logs/lam${lam}_lip${lip}.out"

        echo "[$(date +%H:%M:%S)] Running: λ=$lam, Lip=$lip"

        python run_experiment.py \
            --problem quadratic \
            --epochs "$epochs" \
            --lam "$lam" \
            --lip-scale "$lip" \
            --quad-features \
            --n-steps 40 \
            --vel-layers 4 \
            --vel-hidden 256 \
            --output-dir "$output_dir" \
            --device cuda \
            --seed 42 \
            > "$log_file" 2>&1

        # Extract final distance from log
        dist=$(grep "mean_dist" "$log_file" | tail -4 | awk '{sum+=$3; count+=1} END {print sum/count}')
        echo "  → Mean distance: $dist"
        echo ""
    done
done

echo "Sweep complete!"
echo "Results: $output_dir"
echo ""
echo "To find best run:"
echo "  grep 'Mean distance' $output_dir/logs/*.out | sort -t':' -k3 -n | head -5"
