#!/bin/bash
# Formulation comparison with smaller eta=0.01
PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"

declare -A CHECKPOINTS
CHECKPOINTS["circle"]="results/circle/checkpoints/it20000_lam0.01_gp1.0_v3x128_st10/checkpoint_iter20000.pt"
CHECKPOINTS["bimodal_quadratic"]="results/bimodal_quadratic/checkpoints/it20000_lam0.01_gp1.0_v4x256_st10_quad/checkpoint_iter20000.pt"

DISC_STEPS=3
DISC_LR=0.001
GP_WEIGHT=0.0
N_TRAIN=10000
BATCH_SIZE=256
K=500
L=1.0
ETA=0.01

for PROBLEM in circle bimodal_quadratic; do
    CKPT="${CHECKPOINTS[$PROBLEM]}"

    if [ "$PROBLEM" = "circle" ]; then
        OUT_DIR="results/circle/gpa_plots/03_formulation"
    else
        OUT_DIR="results/bimodal_quadratic/gpa_plots/03_formulation"
    fi

    for FORMULATION in LT LT_nu DV; do
        # === Flow-start ===
        JOB_NAME="gpa_form2_${FORMULATION}_${PROBLEM}_fresh"
        echo "Submitting: $JOB_NAME"
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
    --problem ${PROBLEM} \
    --no-warmstart \
    --K ${K} \
    --eta ${ETA} \
    --disc-steps ${DISC_STEPS} \
    --disc-lr ${DISC_LR} \
    --L ${L} \
    --gp-weight ${GP_WEIGHT} \
    --formulation ${FORMULATION} \
    --n-train ${N_TRAIN} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUT_DIR}
EOF

        # === Prior-start ===
        JOB_NAME="gpa_form2_${FORMULATION}_${PROBLEM}_prior"
        echo "Submitting: $JOB_NAME"
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
    --problem ${PROBLEM} \
    --from-prior \
    --no-warmstart \
    --K ${K} \
    --eta ${ETA} \
    --disc-steps ${DISC_STEPS} \
    --disc-lr ${DISC_LR} \
    --L ${L} \
    --gp-weight ${GP_WEIGHT} \
    --formulation ${FORMULATION} \
    --n-train ${N_TRAIN} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUT_DIR}
EOF
    done
done

echo "All formulation sweep (eta=0.01) jobs submitted!"
