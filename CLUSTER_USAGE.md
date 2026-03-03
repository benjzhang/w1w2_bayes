# Running on Cluster

## Quick Start

```bash
# Single run
python run_experiment.py --problem quadratic --epochs 300 --device cuda

# Run in background (keeps running after logout)
nohup bash run_quadratic_sweep.sh &

# Or with screen/tmux
screen -S w1w2_sweep
bash run_quadratic_sweep.sh
# Ctrl+A, D to detach
```

## Example Sweep Scripts

### `sweep_example.sh`
General template for parameter sweeps. Edit the hyperparameter arrays:
```bash
problems=("quadratic")
lam_values=("0.001" "0.01" "0.05")
lip_values=("5.0" "10.0" "20.0")
```

Run:
```bash
bash sweep_example.sh
```

Outputs:
- `experiments/sweep_YYYYMMDD_HHMMSS/`
  - `quadratic/` - models and plots
  - `logs/` - stdout/stderr for each run

### `run_quadratic_sweep.sh`
Focused sweep for quadratic problem with best-known architecture (4×256 network, 40 steps).

Run:
```bash
bash run_quadratic_sweep.sh
```

Find best run:
```bash
grep 'Mean distance' experiments/quadratic_sweep_*/logs/*.out | sort -t':' -k3 -n | head -5
```

## Cluster-Specific Tips

### GPU Selection
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
python run_experiment.py --problem quadratic --device cuda
```

### Background Running
```bash
# nohup (simple)
nohup bash run_quadratic_sweep.sh > sweep.log 2>&1 &
tail -f sweep.log  # Monitor progress

# screen (recommended)
screen -S experiment
bash run_quadratic_sweep.sh
# Ctrl+A, D to detach
# screen -r experiment  # Reattach later

# tmux
tmux new -s experiment
bash run_quadratic_sweep.sh
# Ctrl+B, D to detach
```

### Monitoring Progress
```bash
# Watch log file
tail -f experiments/*/logs/lam0.01_lip10.0.out

# Check running jobs
ps aux | grep run_experiment

# GPU usage
nvidia-smi -l 1
```

## Output Structure

After a sweep, you'll have:
```
experiments/quadratic_sweep_20260303_123456/
├── logs/                           # Stdout logs
│   ├── lam0.001_lip5.0.out
│   ├── lam0.001_lip10.0.out
│   └── ...
└── quadratic/                      # Results
    ├── model_ep300_lam0.001_lip5.0_v4x256_st40_quad.pt
    ├── results_ep300_lam0.001_lip5.0_v4x256_st40_quad.png
    └── checkpoints/
        └── ep300_lam0.001_lip5.0_v4x256_st40_quad/
```

## Finding Best Hyperparameters

```bash
# Extract mean distances from all logs
cd experiments/quadratic_sweep_YYYYMMDD_HHMMSS/logs
for f in *.out; do
    echo -n "$f: "
    grep "mean_dist" "$f" | tail -4 | awk '{sum+=$3; n++} END {print sum/n}'
done | sort -t':' -k2 -n | head -5

# Or use this one-liner
grep -h "y=.*mean_dist" *.out | \
    awk '{sum+=$3; n++} END {print sum/n}' | \
    paste <(ls *.out) - | \
    sort -k2 -n | head -5
```

## Common Workflows

### Test run (fast)
```bash
python run_experiment.py --problem linear --epochs 10 --device cuda
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
