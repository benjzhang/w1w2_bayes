#!/bin/bash
# Test hard spectral norm projection (matching reference GPA implementation)
# Key: gp_weight=0 triggers hard projection, sweep L values
PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"

declare -A CHECKPOINTS
CHECKPOINTS["circle"]="results/circle/checkpoints/it20000_lam0.01_gp1.0_v3x128_st10/checkpoint_iter20000.pt"
CHECKPOINTS["bimodal_quadratic"]="results/bimodal_quadratic/checkpoints/it20000_lam0.01_gp1.0_v4x256_st10_quad/checkpoint_iter20000.pt"

# Match reference: disc_steps=3, eta=0.5, no GP (hard spec norm)
DISC_STEPS=3
DISC_LR=0.001
GP_WEIGHT=0.0
N_TRAIN=10000
BATCH_SIZE=256

# Sweep L and K
L_VALUES=(1.0 10.0 100.0)
K_VALUES=(500 1000)
ETA=0.5

for PROBLEM in circle bimodal_quadratic; do
    CKPT="${CHECKPOINTS[$PROBLEM]}"

    if [ "$PROBLEM" = "circle" ]; then
        OUT_DIR="results/circle"
    elif [ "$PROBLEM" = "bimodal_quadratic" ]; then
        OUT_DIR="results/bimodal_quadratic"
    fi

    for L in "${L_VALUES[@]}"; do
        for K in "${K_VALUES[@]}"; do
            if [ "$K" -le 500 ]; then
                TIME="01:00:00"
            else
                TIME="02:00:00"
            fi

            # === Flow-start + fresh disc + hard spec norm ===
            JOB_NAME="gpa_sn_${PROBLEM}_L${L}_K${K}_fresh"
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

            # === Prior + fresh disc + hard spec norm ===
            JOB_NAME="gpa_sn_${PROBLEM}_L${L}_K${K}_prior"
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
done

echo "All spec norm jobs submitted!"
