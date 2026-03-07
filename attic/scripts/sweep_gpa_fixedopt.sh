#!/bin/bash
# Test with fixed persistent optimizer: L=10, L=100, L=1000
# Compare flow-start vs prior for circle, K=500, eta=0.01
PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"
CKPT="results/circle/checkpoints/it20000_lam0.01_gp1.0_v3x128_st10/checkpoint_iter20000.pt"
OUT_DIR="results/circle"

DISC_STEPS=10
DISC_LR=0.001
GP_WEIGHT=1.0
N_TRAIN=10000
BATCH_SIZE=256
K=500
ETA=0.01

for L in 10.0 100.0 1000.0; do
    for MODE in fresh prior; do
        JOB_NAME="gpa_fixopt_circle_L${L}_${MODE}"
        echo "Submitting: $JOB_NAME"

        EXTRA_ARGS=""
        if [ "$MODE" = "prior" ]; then
            EXTRA_ARGS="--from-prior --no-warmstart"
        else
            EXTRA_ARGS="--no-warmstart"
        fi

        sbatch <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o ${PROJECT_DIR}/logs/slurm/${JOB_NAME}_%j.out
#SBATCH -e ${PROJECT_DIR}/logs/slurm/${JOB_NAME}_%j.err

cd "$PROJECT_DIR"
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes
export PYTHONUNBUFFERED=1

python -u run_gpa_refine.py \
    --checkpoint ${CKPT} \
    --problem circle \
    ${EXTRA_ARGS} \
    --K ${K} \
    --eta ${ETA} \
    --disc-steps ${DISC_STEPS} \
    --disc-lr ${DISC_LR} \
    --L ${L} \
    --gp-weight ${GP_WEIGHT} \
    --n-train ${N_TRAIN} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUT_DIR}
EOF
    done
done

echo "All fixed-optimizer jobs submitted!"
