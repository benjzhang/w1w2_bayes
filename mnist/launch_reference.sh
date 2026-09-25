#!/bin/bash
# Run the reference W1W2 flow implementation on MNIST (unconditional generation).
# This runs the original TF code from HyeminGu's repo to verify it works.

set -e

SCRIPT=$(mktemp /tmp/ref_mnist_XXXX.sh)

cat > ${SCRIPT} << 'EOF'
#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=l40s|a100|h100|a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ref_mnist
#SBATCH --output=/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/ref_mnist_%j.out
#SBATCH --error=/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs/ref_mnist_%j.err

set -e
export PYTHONUNBUFFERED=1

source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes

cd /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_reference

echo "=== Reference MNIST (unconditional) ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start: $(date)"

python -c "import tensorflow as tf; print('TF version:', tf.__version__)"

# Run with their default MNIST settings from the paper:
# lr_gen=5e-4, lr_disc=1e-4, L=1.0, alpha1=0.05, T=5.0, dt=1.0, 5000 iters
# f=reverse_KL, loss_case=OT, gradient penalty lambda=0.1
python POT_Flow_GAN_CNN.py \
    --example MNIST \
    --total_dim 784 \
    --Rep 3 \
    --f reverse_KL \
    -L 1.0 \
    --alpha1 0.05 \
    --T 5.0 \
    --dt 1.0 \
    --loss_case OT \
    --iterations 5000 \
    --lr_gen 5e-4 \
    --lr_disc 1e-4 \
    --lamda 0.1 \
    --gen_dims 512 512 512 \
    --disc_dims 256 256 256 \
    --conv_layer_channels 8 8

echo "Done: $(date)"
EOF

mkdir -p /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes/logs
JID=$(sbatch ${SCRIPT} | awk '{print $4}')
echo "Submitted reference MNIST job -> ${JID}"
echo "Log: logs/ref_mnist_${JID}.out"
rm ${SCRIPT}
