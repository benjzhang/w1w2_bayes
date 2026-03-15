#!/bin/bash
# Re-run circle baselines with y=0.25 added
set -e
mkdir -p logs/slurm

PROBLEM=circle
JOB_NAME="baselines_14_circle_025"
OUT_DIR="results/circle/14_sweep/baselines"

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

python -u run_baselines.py \
    --problem circle \
    --n-iters 20000 \
    --hidden 128 \
    --n-layers 3 \
    --n-train 10000 \
    --n-eval 2000 \
    --n-steps 10 \
    --sgm-sample-steps 200 \
    --flow-lam 0.25 \
    --flow-lip-scale 10.0 \
    --flow-gp-lambda 1.0 \
    --gpa-K 500 \
    --gpa-eta 0.005 \
    --gpa-L 1000 \
    --gpa-disc-steps 10 \
    --gpa-disc-hidden 32 \
    --gpa-disc-layers 4 \
    --cnf-lam 0.01 \
    --output-dir $OUT_DIR \
    --seed 42

echo "Done: circle with y=0.25"
EOF
