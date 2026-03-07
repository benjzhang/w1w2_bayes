#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mnist_%j.out
#SBATCH --error=logs/mnist_%j.err

module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes

cd /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes

mkdir -p logs results/mnist

python -m mnist.train_flow --n-iters 20000 --output-dir results/mnist
python -m mnist.gpa_refine --checkpoint results/mnist/model.pt --output-dir results/mnist
python -m mnist.evaluate --checkpoint results/mnist/model.pt --output-dir results/mnist
