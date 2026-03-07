#!/bin/bash
# GPA refinement starting from lam=0.1, lip=10 flow (more diffuse particles)
# Compare hard SN vs GP, flow-start vs prior
PROJECT_DIR="/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes"

CKPT_CIRCLE="results/circle/checkpoints/it20000_lam0.1_gp10.0_v3x128_st10/checkpoint_iter20000.pt"
CKPT_BIMODAL="results/bimodal_quadratic/checkpoints/it20000_lam0.1_lip10.0_v4x256_st10_quad/checkpoint_iter20000.pt"

N_TRAIN=10000
BATCH_SIZE=256
K=500

# Hard SN: ds=3, ReLU, gp_weight=0
# GP: ds=10, SiLU, gp_weight=1.0

for PROBLEM in circle bimodal_quadratic; do
    if [ "$PROBLEM" = "circle" ]; then
        CKPT="$CKPT_CIRCLE"
        OUT_DIR="results/circle"
    else
        CKPT="$CKPT_BIMODAL"
        OUT_DIR="results/bimodal_quadratic"
    fi

    # --- Hard SN runs ---
    for L in 1.0 10.0 100.0; do
        for ETA in 0.01 0.05; do
            # Flow-start
            JOB_NAME="gpa_lam01_sn_${PROBLEM}_L${L}_eta${ETA}_fresh"
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
    --disc-steps 3 \
    --disc-lr 0.001 \
    --L ${L} \
    --gp-weight 0.0 \
    --n-train ${N_TRAIN} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUT_DIR}
EOF
        done
    done

    # --- GP runs (for comparison) ---
    for ETA in 0.01 0.1; do
        # Flow-start with GP
        JOB_NAME="gpa_lam01_gp_${PROBLEM}_eta${ETA}_fresh"
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
    --disc-steps 10 \
    --disc-lr 0.001 \
    --L 10.0 \
    --gp-weight 1.0 \
    --n-train ${N_TRAIN} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUT_DIR}
EOF
    done
done

echo "All lam=0.1 jobs submitted!"
