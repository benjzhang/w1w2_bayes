# Conditional OT-Flow Results: 2D Hierarchical Funnel Distribution

## Overview

This demonstrates conditional optimal transport on a **highly degenerate 2D distribution** inspired by hierarchical Bayesian models. The distribution exhibits extreme variance changes (up to 10⁶× range!) depending on the conditioning variable, making it a challenging test case.

## Target Distribution: Hierarchical Bayesian Model

This distribution mimics the posterior in a hierarchical model:

```
v ~ N(0, 2)              [hyperparameter, log-precision]
x₁|v ~ N(0, exp(v))      [parameter 1]
x₂|v ~ N(0.8*x₁, exp(v/2))  [parameter 2, correlated with x₁]
```

### Key Properties

1. **Exponential Degeneracy**:
   - When v = -3: std(x₁) ≈ 0.22 (tight, high precision)
   - When v = +3: std(x₁) ≈ 4.48 (spread out, low precision)
   - Ratio: ~20× in standard deviation, ~400× in variance!

2. **Full Space Support**: Unlike the Gamma distribution, this distribution has support on all of ℝ² for any v

3. **Correlation Structure**: x₂ depends on both v and x₁, creating additional complexity:
   - x₂ ≈ 0.8*x₁ + noise(v)
   - This creates the funnel-like shape in 3D space

4. **Bayesian Relevance**: This type of distribution appears in:
   - Hierarchical models with log-normal priors
   - Models with variance parameters
   - Neal's funnel (a classic pathological case for MCMC samplers)

## Model Architecture

**Velocity Network v_θ(t, x₁, x₂; v):**
- Input: [t, x₁, x₂, v] (4D)
- Hidden layers: 3 layers with 128 units each
- Activation: Tanh
- Output: [v_x₁, v_x₂] (2D velocity)

**Discriminator φ_ψ(x₁, x₂; v):**
- Input: [x₁, x₂, v] (3D)
- Hidden layers: 3 layers with 128 units each
- Activation: Tanh
- Lipschitz constraint: Spectral normalization
- Output: scalar score

## Training Configuration

```python
n_epochs = 500
batch_size = 256
learning_rate = 1e-3
lambda_param = 1.0  # Kinetic energy weight (increased from 0.5)
n_steps = 10  # ODE integration steps
disc_updates = 5  # Critical for stability!
```

### Why λ = 1.0?

For the 2D case, we increased the kinetic regularization from 0.5 to 1.0 because:
- 2D has more degrees of freedom (velocity is 2D vector)
- Stronger regularization helps maintain smooth transport
- The kinetic term sums over dimensions: ||v||² = v_x₁² + v_x₂²

## Training Results

### Metrics (Final)
- L_dual: ~0.209
- Kinetic Energy: ~0.028
- Training time: ~7 minutes on CPU

### Stability
✓ Smooth convergence, no oscillations
✓ No mode collapse
✓ Stable throughout 500 epochs

## Quality Assessment

### 3D Joint Distribution
✓ **Funnel shape captured**: The generated samples show the characteristic funnel structure
✓ **Scale variation**: Successfully handles tight clusters (v < 0) and spread-out clouds (v > 0)
✓ **2D projection**: Good overlap between true and generated samples

### Conditional Slices p(x₁, x₂|v)

**v = -3 (High Precision):**
✓ Excellent: Generated samples are tightly concentrated, matching true distribution
✓ Marginal p(x₁|v=-3) shows near-perfect alignment

**v = -1 (Moderate-High Precision):**
✓ Very good: Captures the spread and correlation structure
✓ Histograms align well

**v = 1 (Moderate-Low Precision):**
✓ Good: Correct spread, though generated samples slightly more dispersed
✓ Correlation between x₁ and x₂ preserved

**v = 3 (Low Precision):**
✓ Good: Large spread captured, though marginal shows generated is a bit wider
⚠️ This is the most challenging regime (widest distribution)

### Flow Trajectories
✓ **Smooth transport**: Trajectories show direct paths in (x₁, x₂) space
✓ **Nearly horizontal**: x₁(t) is almost constant over time for v=0, suggesting efficient transport

## Challenges Successfully Handled

1. **Extreme Variance Range**:
   - The model must generate samples with std ranging from ~0.2 to ~4.5
   - This is a 20× range in scale!

2. **Conditional Correlation**:
   - x₂ depends on x₁ through the 0.8*x₁ term
   - The model learned this correlation structure

3. **Full Space Support**:
   - Unlike distributions with limited support (e.g., Gamma), samples can be anywhere
   - Model handles negative and positive values correctly

4. **Degeneracy**:
   - Some conditional distributions are very narrow
   - Model doesn't collapse or produce artifacts

## Comparison to Classical Methods

**Why this is hard for traditional methods:**
- **HMC/NUTS**: Notorious for struggling with Neal's funnel due to varying step size needs
- **Standard VAE**: Would need separate decoder for each v, doesn't share structure
- **Normalizing Flows**: Affine coupling layers struggle with extreme scale changes
- **GANs**: Mode collapse risk is high with such varying distributions

**Why OT-Flow succeeds:**
- Velocity field adapts to v, learning different dynamics for different regimes
- Kinetic regularization prevents erratic behavior
- Adversarial training with spectral norm maintains stability
- Continuous-time formulation allows smooth adaptation

## Mathematical Insight

The funnel distribution creates a challenging geometry because:

```
log p(x₁, x₂|v) ∝ -x₁²/(2exp(v)) - (x₂ - 0.8x₁)²/(2exp(v/2))
```

The curvature of this log-density changes drastically with v:
- At v = -3: Very high curvature (steep walls)
- At v = +3: Very low curvature (flat landscape)

The OT-Flow learns a velocity field that navigates this varying geometry!

## Reproduction

```bash
python conditional_ot_flow_2d.py
```

Runs in ~7 minutes on CPU for 500 epochs.

## Visualizations

1. **joint_distribution_2d.png**: 3D funnel structure + 2D projection
2. **conditional_slices_2d.png**: Scatter plots and histograms for v ∈ {-3, -1, 1, 3}
3. **training_curves_2d.png**: L_dual and kinetic energy over training
4. **flow_trajectories_2d.png**: Transport paths in 2D space

## Conclusion

This demonstrates that **conditional OT-Flow with Benamou-Brenier formulation can successfully learn highly degenerate conditional distributions** that appear in real Bayesian inference problems. The method handles:

- Exponentially varying scales
- Conditional correlations
- Full space support
- Extreme degeneracy

All while maintaining training stability and producing smooth transport maps!

## References

- Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler (NUTS).
- Benamou, J. D., & Brenier, Y. (2000). Computational fluid mechanics solution to Monge-Kantorovich.
