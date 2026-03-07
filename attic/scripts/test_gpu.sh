#!/bin/bash
#SBATCH --job-name=w1w2_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/%x_%j.out
#SBATCH --error=/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm/%x_%j.err

# Ensure log directory exists
mkdir -p /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/slurm

module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes

echo "=== GPU Test ==="
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "=== Running test experiment (linear, 10 epochs) ==="
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes
python run_experiment.py --problem linear --epochs 10 --device cuda --output-dir results/test_run

echo ""
echo "=== Done ==="
