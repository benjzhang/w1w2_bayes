#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mnist_sweep_%j.out
#SBATCH --error=logs/mnist_sweep_%j.err
#SBATCH --job-name=mnist_sweep

set -e
source /cache/home/bj394/w1w2_bayes/mnist/_env.sh

echo "=== MNIST Inpainting Sweep ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "Start:  $(date)"
echo ""

python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"

python -m mnist.sweep --output-dir results/mnist/sweep --n-test 10 --n-eval-samples 64

echo ""
echo "Done: $(date)"
