#!/bin/bash
#
# Quadratic problem: sweep over λ and Lip scale
# Finds best hyperparameters for the parabola constraint
#

problems=("quadratic")
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
