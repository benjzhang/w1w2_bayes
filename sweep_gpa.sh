#!/bin/bash
# Sweep GPA refinement across problems and hyperparameters
# Submit with: bash sweep_gpa.sh

PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"

# Problems and their best checkpoints
declare -A CHECKPOINTS
CHECKPOINTS["circle"]="results/circle/checkpoints/it20000_lam0.01_gp1.0_v3x128_st10/checkpoint_iter20000.pt"
CHECKPOINTS["bimodal_quadratic"]="results/bimodal_quadratic/checkpoints/it20000_lam0.01_gp1.0_v4x256_st10_quad/checkpoint_iter20000.pt"

# Hyperparameter sweep
K_VALUES=(500 1000 2000)
ETA_VALUES=(0.01 0.1 0.5)

# Fixed params
DISC_STEPS=20
DISC_LR=0.001
L=1.0
GP_WEIGHT=1.0
N_TRAIN=10000
BATCH_SIZE=256

for PROBLEM in circle bimodal_quadratic; do
    CKPT="${CHECKPOINTS[$PROBLEM]}"

    # Set output dir based on problem
    if [ "$PROBLEM" = "circle" ]; then
        OUT_DIR="results/circle"
    elif [ "$PROBLEM" = "bimodal_quadratic" ]; then
        OUT_DIR="results/bimodal_quadratic"
    fi

    for K in "${K_VALUES[@]}"; do
        for ETA in "${ETA_VALUES[@]}"; do
            JOB_NAME="gpa_${PROBLEM}_K${K}_eta${ETA}"

            # Time limit: scale with K
            if [ "$K" -le 500 ]; then
                TIME="01:00:00"
            elif [ "$K" -le 1000 ]; then
                TIME="02:00:00"
            else
                TIME="04:00:00"
            fi

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
        done
    done
done

echo "All jobs submitted!"
