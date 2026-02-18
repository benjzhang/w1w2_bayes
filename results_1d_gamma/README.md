# Conditional OT-Flow Results: 1D Gamma Distribution

## Overview

This folder contains results from training a conditional optimal transport flow model to learn a 1D conditional distribution using the Benamou-Brenier formulation.

## Target Distribution

**Joint Distribution:** π(x, y) where:
- y ~ Uniform(-3, 3)
- x|y ~ Gamma(shape=2, scale=0.3) + tanh(y)

This creates a conditional distribution that shifts based on y (via tanh) and maintains a right-skewed Gamma shape.

## Model Architecture

**Velocity Network v_θ(t, x; y):**
- Input: [t, x, y] (3D)
- Hidden layers: 2 layers with 64 units each
- Activation: Tanh
- Output: velocity (1D scalar)

**Discriminator φ_ψ(x; y):**
- Input: [x, y] (2D)
- Hidden layers: 2 layers with 64 units each
- Activation: Tanh
- Lipschitz constraint: Spectral normalization on all layers
- Output: scalar score

## Training Configuration

```python
n_epochs = 500
batch_size = 256
learning_rate = 1e-3
lambda_param = 0.5  # Kinetic energy weight
n_steps = 10  # ODE integration steps
disc_updates = 5  # Discriminator updates per generator update
```

## Training Objective

```
min_θ max_ψ J(θ, ψ) = E[φ_ψ(x_T; y)] - E[exp(φ_ψ(x_real; y) - 1)] + λ * E[∫ 0.5 * ||v_θ||² dt]
```

Where:
- First term: Terminal cost (adversarial)
- Second term: f-divergence with f*(z) = exp(z-1) (KL divergence)
- Third term: Kinetic energy regularization (Benamou-Brenier)

## Key Hyperparameter: Discriminator-to-Generator Ratio

**Critical Finding:** Using 5 discriminator updates per 1 generator update was essential for success.

- With 1:1 ratio → Generated distributions were too broad and diffuse
- With 5:1 ratio → Generated distributions closely match target distributions

This is because the discriminator needs to be well-trained to provide reliable gradients for the velocity network.

## Results

### Training Metrics (Final)
- L_dual: ~0.102
- Kinetic Energy: ~0.320
- Training time: ~5 minutes on CPU

### Quality Assessment
✓ **Excellent match** on conditional histograms for all y values tested
✓ **Smooth transport** trajectories from Gaussian to target
✓ **Stable training** with no mode collapse
✓ **Velocity magnitude** reasonably constant (~0.58)

## Visualizations

1. **joint_distribution.png**: Scatter plot of true vs generated joint distribution
2. **conditional_slices.png**: Histograms comparing p(x|y) for y ∈ {-2, 0, 1, 2}
3. **training_curves.png**: L_dual and kinetic energy over training epochs
4. **flow_trajectories.png**: ODE trajectories showing transport from t=0 to t=1
5. **velocity_magnitude.png**: Mean velocity magnitude over time

## Reproduction

To reproduce these results:

```bash
python conditional_ot_flow.py
```

The script will:
1. Generate 10,000 samples (80% train, 20% validation)
2. Train for 500 epochs with the configuration above
3. Save all visualization plots to the current directory

## Dependencies

```
torch
numpy
matplotlib
seaborn
tqdm
```

## References

- Benamou, J. D., & Brenier, Y. (2000). A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem.
- Chen, Y., Georgiou, T. T., & Pavon, M. (2021). On the relation between optimal transport and Schrödinger bridges.
- Lipman, Y., et al. (2023). Flow Matching for Generative Modeling.
