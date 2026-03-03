#!/bin/bash
#
# Example parameter sweep for W1W2 flow experiments
# Usage: bash sweep_example.sh
#

# Define hyperparameter grids
problems=("quadratic")
lam_values=("0.001" "0.01" "0.05")
lip_values=("5.0" "10.0" "20.0")
epochs=500

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

            echo "Running: problem=$problem lam=$lam lip=$lip -> $log_file"

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

            echo "Completed: problem=$problem lam=$lam lip=$lip"
            echo ""
        done
    done
done

echo "Sweep complete! Results in: $output_dir"
echo "Logs saved to: $output_dir/logs/"
