#!/bin/bash
# Launch GPA refinement on a trained flow checkpoint.
#
# Usage: bash mnist/launch_gpa.sh

set -e
cd /cache/home/bj394/w1w2_bayes
mkdir -p logs

SWEEP_DIR="results/mnist/sweep"
CHECKPOINT="${SWEEP_DIR}/rkl_gp_5k/model.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi
echo "Using checkpoint: ${CHECKPOINT}"

SCRIPT_DIR=$(mktemp -d)
trap "rm -rf ${SCRIPT_DIR}" EXIT

submit_gpa() {
    local NAME=$1
    local EXTRA_ARGS=$2

    cat > ${SCRIPT_DIR}/${NAME}.sh << 'HEADER'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
HEADER
    cat >> ${SCRIPT_DIR}/${NAME}.sh << EOF
#SBATCH --job-name=mnist_${NAME}
#SBATCH --output=logs/mnist_${NAME}_%j.out
#SBATCH --error=logs/mnist_${NAME}_%j.err

set -e
export PYTHONUNBUFFERED=1
source /cache/home/bj394/w1w2_bayes/mnist/_env.sh

echo "=== ${NAME} ==="
echo "Job: \$SLURM_JOB_ID  Node: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
echo "Start: \$(date)"
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"

python -m mnist.run_one gpa --name ${NAME} \
    --checkpoint ${CHECKPOINT} \
    --output-dir ${SWEEP_DIR} \
    ${EXTRA_ARGS}

echo "Done: \$(date)"
EOF

    JID=$(sbatch ${SCRIPT_DIR}/${NAME}.sh | awk '{print $4}')
    echo "  Submitted ${NAME} -> job ${JID}"
}

echo "=== Submitting GPA refinement jobs ==="

# K=1000 quick runs
submit_gpa "gpa_v2_kl_1k" \
    "--K 1000 --eta 0.5 --L 1.0 --n-samples 64 --disc-steps 5 --disc-lr 5e-4 --batch-size 640 --fstar kl --lip-mode project"

submit_gpa "gpa_v2_rkl_1k" \
    "--K 1000 --eta 0.5 --L 1.0 --n-samples 64 --disc-steps 5 --disc-lr 5e-4 --batch-size 640 --fstar reverse_kl --lip-mode gp --gp-lambda 0.1"

# K=5000 longer runs
submit_gpa "gpa_v2_kl_5k" \
    "--K 5000 --eta 0.5 --L 1.0 --n-samples 64 --disc-steps 5 --disc-lr 5e-4 --batch-size 640 --fstar kl --lip-mode project"

submit_gpa "gpa_v2_rkl_5k" \
    "--K 5000 --eta 0.5 --L 1.0 --n-samples 64 --disc-steps 5 --disc-lr 5e-4 --batch-size 640 --fstar reverse_kl --lip-mode gp --gp-lambda 0.1"

echo ""
echo "GPA jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in ${SWEEP_DIR}/gpa_*/"
