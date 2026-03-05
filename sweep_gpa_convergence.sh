#!/bin/bash
# Sweep GPA refinement: flow+warm-start vs prior+fresh across small K values
# Shows advantage of flow initialization + trained discriminator at low compute
# Submit with: bash sweep_gpa_convergence.sh

PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"

# Problems and their best checkpoints
declare -A CHECKPOINTS
CHECKPOINTS["circle"]="results/circle/checkpoints/it20000_lam0.01_gp1.0_v3x128_st10/checkpoint_iter20000.pt"
CHECKPOINTS["bimodal_quadratic"]="results/bimodal_quadratic/checkpoints/it20000_lam0.01_gp1.0_v4x256_st10_quad/checkpoint_iter20000.pt"

# K values: small to moderate
K_VALUES=(10 25 50 100 200 500)

# Fixed params — small eta for stability
ETA=0.01
DISC_STEPS=10
DISC_LR=0.001
L=10.0
GP_WEIGHT=1.0
N_TRAIN=10000
BATCH_SIZE=256

for PROBLEM in circle bimodal_quadratic; do
    CKPT="${CHECKPOINTS[$PROBLEM]}"

    if [ "$PROBLEM" = "circle" ]; then
        OUT_DIR="results/circle"
    elif [ "$PROBLEM" = "bimodal_quadratic" ]; then
        OUT_DIR="results/bimodal_quadratic"
    fi

    for K in "${K_VALUES[@]}"; do
        if [ "$K" -le 50 ]; then
            TIME="00:20:00"
        elif [ "$K" -le 200 ]; then
            TIME="00:40:00"
        else
            TIME="01:30:00"
        fi

        # === Flow output + warm-start disc (NO --no-warmstart) ===
        JOB_NAME="gpa_conv_${PROBLEM}_K${K}_warm"
        echo "Submitting: $JOB_NAME"
        sbatch <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p gpu
#SBATCH -t ${TIME}
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
    --problem ${PROBLEM} \
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

        # === Prior + fresh disc ===
        JOB_NAME="gpa_conv_${PROBLEM}_K${K}_prior"
        echo "Submitting: $JOB_NAME"
        sbatch <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p gpu
#SBATCH -t ${TIME}
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
    --problem ${PROBLEM} \
    --from-prior \
    --no-warmstart \
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

echo "All convergence sweep jobs submitted!"
