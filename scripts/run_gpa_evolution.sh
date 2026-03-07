#!/bin/bash
#SBATCH --job-name=gpa_evo
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/gpa_evo_%A_%a.out
#SBATCH --error=logs/slurm/gpa_evo_%A_%a.err
#SBATCH --array=0-1

set -e
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes
export PYTHONUNBUFFERED=1

PROBLEMS=(circle bimodal_quadratic)
LAMS=(0.25 0.25)

PROBLEM=${PROBLEMS[$SLURM_ARRAY_TASK_ID]}
LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}

FLOW_CKPT="results/${PROBLEM}/14_sweep/lam${LAM}/flow_checkpoint.pt"
OUT_DIR="results/${PROBLEM}/14_sweep/evolution"

cd /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes

python -u run_gpa_evolution.py \
    --problem "$PROBLEM" \
    --load-flow "$FLOW_CKPT" \
    --flow-lam "$LAM" \
    --L 1000 --eta 0.005 --K 300 \
    --n-frames 8 \
    --output-dir "$OUT_DIR"

echo "Done: $PROBLEM"
