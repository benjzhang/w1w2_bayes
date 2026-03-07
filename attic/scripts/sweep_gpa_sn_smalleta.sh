#!/bin/bash
# Hard spectral norm with smaller eta to match gradient scale
PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"

declare -A CHECKPOINTS
CHECKPOINTS["circle"]="results/circle/checkpoints/it20000_lam0.01_gp1.0_v3x128_st10/checkpoint_iter20000.pt"
CHECKPOINTS["bimodal_quadratic"]="results/bimodal_quadratic/checkpoints/it20000_lam0.01_gp1.0_v4x256_st10_quad/checkpoint_iter20000.pt"

DISC_STEPS=3
DISC_LR=0.001
GP_WEIGHT=0.0
N_TRAIN=10000
BATCH_SIZE=256

# Sweep: L * eta should be roughly constant
# L=1 eta=0.5, L=10 eta=0.05, L=100 eta=0.005
declare -A ETAS
ETAS["1.0"]="0.5 0.1 0.01"
ETAS["10.0"]="0.05 0.01"
ETAS["100.0"]="0.005 0.001"

K=500

for PROBLEM in circle bimodal_quadratic; do
    CKPT="${CHECKPOINTS[$PROBLEM]}"

    if [ "$PROBLEM" = "circle" ]; then
        OUT_DIR="results/circle"
    elif [ "$PROBLEM" = "bimodal_quadratic" ]; then
        OUT_DIR="results/bimodal_quadratic"
    fi

    for L in 1.0 10.0 100.0; do
        for ETA in ${ETAS[$L]}; do
            # === Prior + fresh disc + hard spec norm ===
            JOB_NAME="gpa_sn2_${PROBLEM}_L${L}_eta${ETA}_prior"
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
    --n-train ${N_TRAIN} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUT_DIR}
EOF

            # === Flow-start + fresh disc + hard spec norm ===
            JOB_NAME="gpa_sn2_${PROBLEM}_L${L}_eta${ETA}_fresh"
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
    --n-train ${N_TRAIN} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUT_DIR}
EOF
        done
    done
done

echo "All jobs submitted!"
