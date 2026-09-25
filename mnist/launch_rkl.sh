#!/bin/bash
# Launch reverse-KL f-star experiments.
set -e
cd /cache/home/bj394/w1w2_bayes
mkdir -p logs results/mnist/sweep

SCRIPT_DIR=$(mktemp -d)
trap "rm -rf ${SCRIPT_DIR}" EXIT

STEPS=10
BS=200

echo "=== Submitting reverse-KL f-star jobs ==="
JOBIDS=""

submit_job() {
    local NAME=$1
    local EXTRA_ARGS=$2

    cat > ${SCRIPT_DIR}/${NAME}.sh << 'HEADER'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
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

python -m mnist.run_one flow --name ${NAME} \
    --n-steps ${STEPS} --batch-size ${BS} \
    ${EXTRA_ARGS}

echo "Done: \$(date)"
EOF

    JID=$(sbatch ${SCRIPT_DIR}/${NAME}.sh | awk '{print $4}')
    echo "  Submitted ${NAME} -> job ${JID}"
    JOBIDS="${JOBIDS}:${JID}"
}

# lam=0.2, L=1, gp=0.1 — modest increase in KE reg, keep GP light
submit_job "rkl_lam02_gp01" \
    "--arch mlp --hidden 512 --n-layers 4 --lam 0.2 --lip-scale 1.0 --fstar reverse_kl --lip-mode gp --gp-lambda 0.1 --lr 5e-4 --lr-disc 1e-4 --n-iters 10000"

# lam=0.2, L=1, gp=0.2 — slightly stronger GP to help diversity
submit_job "rkl_lam02_gp02" \
    "--arch mlp --hidden 512 --n-layers 4 --lam 0.2 --lip-scale 1.0 --fstar reverse_kl --lip-mode gp --gp-lambda 0.2 --lr 5e-4 --lr-disc 1e-4 --n-iters 10000"

echo ""
echo "All jobs submitted. Job IDs: ${JOBIDS#:}"
echo "To monitor: squeue -u \$USER"
