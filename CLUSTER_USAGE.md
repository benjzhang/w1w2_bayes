# Running on Cluster

## Quick Start

```bash
# Single GPU test (submits to SLURM)
sbatch test_gpu.sh

# Quadratic hyperparameter sweep (12 combos, ~12-24 hours)
sbatch run_quadratic_sweep.sh

# Custom sweep (edit grids first)
sbatch sweep_example.sh

# Check your jobs
squeue -u $USER
```

## Sweep Scripts

### `run_quadratic_sweep.sh`
Focused sweep for quadratic problem with best-known architecture (4x256 network, 40 steps).

Sweeps over:
- lambda: 0.001, 0.005, 0.01, 0.05
- Lip scale: 5.0, 10.0, 20.0
- 12 combinations total, 300 epochs each

Submit:
```bash
sbatch run_quadratic_sweep.sh
```

### `sweep_example.sh`
General template for parameter sweeps. Edit the hyperparameter section at the top:
```bash
problems=("quadratic")
lam_values=("0.001" "0.01" "0.05")
lip_values=("5.0" "10.0" "20.0")
epochs=500
```

Submit:
```bash
sbatch sweep_example.sh
```

### `test_gpu.sh`
Quick sanity check — runs linear problem for 10 epochs to verify GPU works.
```bash
sbatch test_gpu.sh
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch SLURM output in real time
tail -f logs/slurm/quad_sweep_<JOBID>.out

# Watch per-run logs (inside the experiment directory)
tail -f experiments/quadratic_sweep_*/logs/lam0.01_lip10.0.out

# Cancel a job
scancel <JOBID>
```

## Output Structure

SLURM logs go to `logs/slurm/`. Experiment results go to `experiments/`:
```
logs/slurm/
  quad_sweep_12345.out          # SLURM stdout
  quad_sweep_12345.err          # SLURM stderr

experiments/quadratic_sweep_20260303_123456/
  logs/                         # Per-run stdout logs
    lam0.001_lip5.0.out
    lam0.001_lip10.0.out
    ...
  quadratic/                    # Results
    model_ep300_lam0.001_lip5.0_v4x256_st40_quad.pt
    results_ep300_lam0.001_lip5.0_v4x256_st40_quad.png
    checkpoints/
      ep300_lam0.001_lip5.0_v4x256_st40_quad/
```

## Finding Best Hyperparameters

```bash
# Extract mean distances from all logs in a sweep
cd experiments/quadratic_sweep_YYYYMMDD_HHMMSS/logs
for f in *.out; do
    echo -n "$f: "
    grep "mean_dist" "$f" | tail -4 | awk '{sum+=$3; n++} END {print sum/n}'
done | sort -t':' -k2 -n | head -5
```

## Common Workflows

### Single run on GPU
```bash
# Wrap in a SLURM interactive session
srun --partition=gpu --gres=gpu:1 --mem=8G --time=02:00:00 --pty bash
module load conda/latest
conda activate /work/bjzhang_umass_edu/.conda/envs/w1w2_bayes
cd /work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_bayes
python run_experiment.py --problem quadratic --epochs 300 --device cuda
```

### Full training
```bash
python run_experiment.py \
    --problem quadratic \
    --epochs 500 \
    --lam 0.01 \
    --lip-scale 10.0 \
    --quad-features \
    --n-steps 40 \
    --vel-layers 4 \
    --vel-hidden 256 \
    --device cuda \
    --output-dir experiments/best_quadratic
```

### Resume from checkpoint
```bash
python run_experiment.py \
    --problem quadratic \
    --resume experiments/best_quadratic/quadratic/checkpoints/.../checkpoint_epoch200.pt \
    --device cuda
```
