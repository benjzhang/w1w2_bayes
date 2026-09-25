#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mnist_%j.out
#SBATCH --error=logs/mnist_%j.err

set -e
source /cache/home/bj394/w1w2_bayes/mnist/_env.sh

mkdir -p results/mnist

python -m mnist.train_flow --n-iters 20000 --output-dir results/mnist
python -m mnist.gpa_refine --checkpoint results/mnist/model.pt --output-dir results/mnist
python -m mnist.evaluate --checkpoint results/mnist/model.pt --output-dir results/mnist
