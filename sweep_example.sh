#!/bin/bash
#
# Example parameter sweep for W1W2 flow experiments
# Edit the hyperparameter grids below, then submit with: sbatch sweep_example.sh
#
#SBATCH --job-name=w1w2_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
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

# ============================================================
# EDIT THIS SECTION: define your sweep
# ============================================================
problems=("quadratic" "bimodal_quadratic")
lam_values=("0.001" "0.01" "0.05")
lip_values=("5.0" "10.0" "20.0")
epochs=500
# ============================================================

# Output directory
output_dir="experiments/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output_dir/logs"

echo "Starting sweep: $output_dir"
echo "Problems: ${problems[@]}"
echo "Lambda: ${lam_values[@]}"
echo "Lip scales: ${lip_values[@]}"
echo ""

# Loop over hyperparameters
for problem in "${problems[@]}"; do
    for lam in "${lam_values[@]}"; do
        for lip in "${lip_values[@]}"; do
            # Create log filename
            log_file="$output_dir/logs/${problem}_lam${lam}_lip${lip}.out"

            echo "[$(date +%H:%M:%S)] Running: problem=$problem lam=$lam lip=$lip"

            python run_experiment.py \
                --problem "$problem" \
                --epochs "$epochs" \
                --lam "$lam" \
                --lip-scale "$lip" \
                --quad-features \
                --output-dir "$output_dir" \
                --device cuda \
                --seed 42 \
                > "$log_file" 2>&1

            echo "  Completed: problem=$problem lam=$lam lip=$lip"
            echo ""
        done
    done
done

echo "Sweep complete! Results in: $output_dir"
echo "Logs saved to: $output_dir/logs/"
