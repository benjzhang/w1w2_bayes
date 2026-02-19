#!/bin/bash
#SBATCH -J cond_ot_flow_2d        # Job name
#SBATCH -p gpu-preempt            # Partition (gpu-preempt for shorter jobs)
#SBATCH -t 02:00:00               # Time limit (2 hours: env setup + training)
#SBATCH --gpus=1                  # Request 1 GPU
#SBATCH --mem=16G                 # Memory
#SBATCH -c 4                      # CPU cores
#SBATCH -o ot_flow_2d_%j.out      # stdout log
#SBATCH -e ot_flow_2d_%j.err      # stderr log

set -e

ENV_NAME="w1w2_bayes"
ENV_PATH="/work/bjzhang_umass_edu/.conda/envs/${ENV_NAME}"
WORK_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Load and activate conda env (must already exist — run setup_env.sh first)
module load conda/latest
if [ ! -d "$ENV_PATH" ]; then
    echo "ERROR: Conda environment not found at $ENV_PATH"
    echo "Run setup_env.sh first: bash setup_env.sh"
    exit 1
fi
conda activate "$ENV_PATH"

# Verify GPU
echo ""
echo "=== GPU Info ==="
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Run the script
echo ""
echo "Running conditional_ot_flow_2d.py..."
cd "$WORK_DIR"
python conditional_ot_flow_2d.py

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
