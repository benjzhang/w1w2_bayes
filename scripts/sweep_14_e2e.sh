#!/bin/bash
# Sweep 14: End-to-end parameter sweep
#
# Flow settings (fixed across sweep):
#   lip_scale=10, gp_lambda=1.0, disc_hidden=32, disc_layers=4, 20k iters
#
# Sweep axes:
#   flow_lam: 0.01, 0.1, 0.25, 0.5
#   GPA (L, eta): (10, 0.01), (100, 0.01), (1000, 0.005)
#
# One SLURM job per (lam, problem) = 8 parallel jobs
# Each job trains flow once, then sweeps 3 GPA configs
#
# Results saved to: results/{problem}/14_sweep/lam{lam}/
#   - hparams.json          (all hyperparameters)
#   - flow_checkpoint.pt    (reload flow without retraining)
#   - flow_particles_before.npz
#   - particles_*.npz       (before/after per GPA config)
#   - metrics_*.json        (full trajectory per GPA config)
#   - summary.json          (final comparison table)
#   - e2e_*.png, e2e_dist_*.png (plots)

set -e
mkdir -p logs/slurm

GPA_CONFIGS="10:0.01,100:0.01,1000:0.005"

for PROBLEM in circle bimodal_quadratic; do
    for LAM in 0.01 0.1 0.25 0.5; do
        JOB_NAME="e2e_14_${PROBLEM}_lam${LAM}"
        OUT_DIR="results/${PROBLEM}/14_sweep/lam${LAM}"

        echo "Submitting: $JOB_NAME -> $OUT_DIR"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/${JOB_NAME}_%j.out
#SBATCH --error=logs/slurm/${JOB_NAME}_%j.err

set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

python -u run_e2e_sweep.py \
    --problem $PROBLEM \
    --flow-iters 20000 \
    --flow-lam $LAM \
    --flow-lip-scale 10.0 \
    --flow-gp-lambda 1.0 \
    --flow-n-steps 10 \
    --gpa-configs "$GPA_CONFIGS" \
    --K 500 \
    --disc-steps 10 \
    --n-train 10000 \
    --n-eval 2000 \
    --output-dir $OUT_DIR \
    --seed 42

echo "Done: $PROBLEM lam=$LAM"
EOF
    done
done

echo "Submitted all jobs."
