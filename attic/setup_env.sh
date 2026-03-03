#!/bin/bash
# Setup conda environment for conditional OT-Flow
# Run this on a compute node (via srun or sbatch)

set -e

ENV_NAME="w1w2_bayes"
ENV_PATH="/work/bjzhang_umass_edu/.conda/envs/${ENV_NAME}"

# Load conda
module load conda/latest

# Check if env already exists
if [ -d "$ENV_PATH" ]; then
    echo "Environment ${ENV_NAME} already exists at ${ENV_PATH}"
    echo "Activating existing environment..."
    conda activate "$ENV_PATH"
else
    echo "Creating new conda environment: ${ENV_NAME}"
    conda create -y -p "$ENV_PATH" python=3.11
    conda activate "$ENV_PATH"

    echo "Installing PyTorch with CUDA 11.8 support..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    echo "Installing remaining dependencies..."
    conda install -y numpy matplotlib seaborn tqdm scipy -c conda-forge
fi

echo "Environment setup complete!"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
