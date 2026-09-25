# Codebase audit — W1⊕W2 conditional sampling

Read-only review against `MFG_GANs_cond.pdf`. Nothing here has been changed.
Line refs are to the tree at the time of writing.

Each finding is tagged:
- **[verified]** — ran code or executed a check; evidence given
- **[read]** — read directly from source, not executed
- **[reasoning]** — inference from the math; not independently checked

No tradeoff recommendations are made. Where a choice exists, it is listed in §5
without a preferred option.

---

## 1. Structure: three parallel implementations

The same mathematics is implemented three times, and the copies have drifted.

| concept | `flows/` + `refinement/` | `mnist/` | `mnist/sweep.py` |
|---|---|---|---|
| `euler_integrate` | `utils/integrators.py` | `run_one.py:34` | `sweep.py:47` |
| `MollifiedReLU` | `refinement/gpa.py:27` | `networks.py:12` | — |
| `_project_disc_weights` | `refinement/gpa.py:58` | `run_one.py:~95` | `sweep.py:72` |
| dual `L_dual` | `w1w2_flow.py:230`, `gpa.py:228` | `run_one.py:149` **and** `:341` | inline |

`mnist/networks.py:3` states it outright: *"Self-contained: does not import from
parent package."* Every discrepancy below is drift between these copies.

**Dead / superseded modules.** `mnist/train_flow.py` (238 lines) and
`mnist/gpa_refine.py` (323) are reachable only from `mnist/run.sh`; the actual
sweeps use `run_one.py` and `sweep.py`. `mnist/gpa_refine.py:81` defines a second
`gpa_refine` unrelated to `refinement/gpa.py:98`. `utils/integrators.py:38`
`compute_kinetic_energy` is imported nowhere and differs subtly from the inlined
copy in `w1w2_flow.py:259` (it detaches the path; the inlined version does not).

**Missing drivers.** `run_e2e_sweep.py` and `run_baselines.py` produced every
number in `results/` and were **deleted in `c2fd7f4`**. Recoverable via
`git show c2fd7f4^:run_e2e_sweep.py`. As it stands the published results cannot
be reproduced from the tree.

---

## 2. Divergences from the paper

### 2.1 Γ — which function class is searched

`D_f^Γ(ρ_T ‖ π) = sup_{φ∈Γ} { E_{ρ_T}[φ] − E_π[f*∘φ] }` is *defined* by Γ. A
different Γ is a different divergence, independent of any algorithm. Paper §1:
Γ = Lipschitz on ℝ^{n+m}, i.e. jointly in (x,y).

**[read]** The two regularization modes search different classes:

- **projection** (`_project_disc_weights`): bounds `‖W₁‖₂` where `W₁` acts on
  `cat([theta, y])`, and bounds all layers ⇒ the realized class is joint in (x,y),
  a **subset** of Γ (`L^(1/D)` is a conservative bound — see §2.7).
- **gradient penalty**: all three copies penalize `‖∇_x φ‖` only
  (`w1w2_flow.py:131` `inputs=theta_hat`; `gpa.py:246` `interp` from θ only;
  `run_one.py` same), at interpolation points built from θ at fixed y. Nothing
  couples different y ⇒ the realized class is not contained in Γ.

**[reasoning]** Writing Γ_x for "Lipschitz in x for each y, no control across y":
Γ ⊊ Γ_x, so `sup_{Γ_x} ≥ sup_Γ`. Projection lands on a subset of Γ (conservative,
still a valid Lipschitz-regularized divergence with an unknown effective constant);
the current GP searches a superset (admits φ that Eq. (1) does not).

**[reasoning]** This also bears on Eq. (3)→(4), which moves `sup_φ` inside
`∫dπ_Y`. That step relies on the sup over Γ decomposing into per-y sups over
sections that inherit joint control. Without a constraint coupling y, the per-y
problems decouple — a different variational problem in the same notation. I have
not checked whether the paper's derivation elsewhere addresses this.

**Correction to an earlier claim:** a gradient penalty *can* target Γ correctly —
penalizing `‖∇_{(x,y)}φ‖` bounds the joint gradient norm, hence the joint
Lipschitz constant. The defect in the current code is that y is omitted, not that
gradient penalties are unsuited to the task. Joint GP and projection differ in
hard-vs-soft constraint and in conservatism; **no recommendation is made here** —
see §5.1.

This splits by *mode*, not by file — one decision applied in three places.

**[read]** Bearing on published results: the `14_sweep` runs used
`flow_gp_lambda=1.0`, i.e. GP mode.

### 2.2 `flows/` gradient penalty ignores `lip_scale`

**[verified]** `w1w2_flow.py:143` hardcodes the threshold at 1:
`penalty = mean(max(0, ‖∇φ‖² − 1))`. `lip_scale` is not used in GP mode. So
`--flow-lip-scale 10` had no effect on any GP-mode run — the flow was constrained
to Γ₁ regardless. Combined with 2.1, the baseline flows differ from Eq. (3) in
two ways at once.

### 2.3 Three gradient-penalty formulas

| site | formula | at ‖g‖=2, L=1 |
|---|---|---|
| `flows/w1w2_flow.py:143` | `max(0, ‖g‖² − 1)` | 3.0 |
| `refinement/gpa.py:249` | `max(0, ‖g‖ − L)²` | 1.0 |
| `mnist/run_one.py` | `max(0, ‖g‖² − L²)` | 3.0 |

(User has confirmed differing GP forms are acceptable; recorded for completeness.)

### 2.4 Naming: `fstar` vs `formulation`

**[verified]** `mnist`'s `--fstar reverse_kl` computes `E[φ] − log E[exp φ]` — this is the
**Donsker–Varadhan** form of the *same forward KL*, identical to
`refinement/gpa.py`'s `formulation='DV'`. **Neither is reverse KL** (that would
need `f(t) = −log t`, `f*(s) = −1 − log(−s)`, implemented nowhere).

Two vocabularies for one concept; one of them misnames it.

*Why LT is unstable in high dimension:* DV is invariant to `φ → φ + c`; LT is not.
LT attains KL only at `φ* = 1 + log(dP/dQ)` **exactly**, constant included. In 784-d
the critic cannot pin that constant, so `exp(φ−1)` overflows. This is an estimator
property, not a modeling one.

### 2.5 `f` is fixed to KL in the low-dim path

**[read]** Paper writes general `f`. `flows/w1w2_flow.py` hardcodes `f*(t) = exp(t−1)` with no
flag. `LT` / `LT_nu` / `DV` are three *representations* of KL, not different `f`.

### 2.6 λ placement — consistent

Paper `(λ/2)∫‖v‖²`; code `L_dual + lam*KE` with the ½ inside KE
(`w1w2_flow.py:259`). Equivalent. (User: not meaningful.)

### 2.7 Spectral projection ≠ Lipschitz constant

`L^(1/D)` per layer bounds the *product* of spectral norms, an upper bound on the
true Lipschitz constant, loose and compounding with depth. `L` is a budget, not a
realized constant. (User: acceptable heuristic.)

### 2.8 `µ₀ = prior` — no issue for current problems

**[verified]** Both `circle` and `bimodal_quadratic` draw `theta = randn(n,2)`, so the prior *is*
N(0,I) and `µ₀ = N(0,I)` coincides with it. The "GPA-only from prior" control is
correctly labelled. **Will not hold for the Euler problem** (uniform on SO(3) ×
inverse-Wishart) — `--source-samples` becomes mandatory there.

---

## 3. Bugs

### 3.1 CNF-MLE is broken — explains the `NaN` baseline

**[verified]** `baselines/cnf.py:80-86`: the trajectory is advanced under `torch.no_grad()` with
`v_detached`. Verified: `theta_0.requires_grad == False`, `grad_fn is None`. So
`log p₀(θ₀)` contributes **no gradient**; only `delta_logp` does. The model
optimizes the trace term alone and never learns to map data to the reference —
hence `NaN` for every entry in both `results.json` files.

CNF-MLE is currently not a valid baseline.

### 3.2 Silent divergence

**[verified]** No finiteness guard anywhere in `gpa.py` / `w1w2_flow.py`; `pipeline.py:290`
calls `np.savez` unconditionally. A diverged run writes an all-`NaN` file and
exits 0 — indistinguishable from success inside a sweep.

### 3.3 CLI defaults contradict library defaults

| param | `refinement/gpa.py` | `pipeline.py` |
|---|---|---|
| `L` | 1.0 | 1000.0 |
| `eta` | 0.5 | 0.005 |

MNIST launchers always pass `--L 1.0` explicitly, so production runs were unaffected.

### 3.4 Warm-start drops the regularization regime

**[read]** `gpa.py:170-172` deep-copies the flow's disc, sets `lip_scale = 1.0`, then
hard-projects to budget `L`. The flow trained that disc under `gp_lambda=1.0`
(penalty, no projection). Weights are inherited; the regime they were learned
under is not.

---

## 4. Best recorded results

**No written conclusion exists anywhere** — no README, no notes; git messages are
`stuff`, `gif`, `update`. Below is reconstructed from JSON.

### 4.1 Baselines (`results/*/14_sweep/baselines/results.json`)

Mean distance to the true manifold, lower better:

| method | circle | bimodal |
|---|---|---|
| W1W2 alone | 0.0285 | 0.0711 |
| **W1W2+GPA** | **0.0137** | **0.0182** |
| CFM | 0.0469 | 0.1501 |
| SGM | 0.0165 | 0.1245 |
| CNF-MLE | `NaN` | `NaN` (see 3.1) |

Config: 20k iters, `flow_lam=0.25`, `lip_scale=10`, `gp_lambda=1.0`,
`gpa_K=500`, `gpa_eta=0.005`, **`gpa_L=1000.0`**, seed 42.

Caveats: SGM is competitive on circle `dist` (0.0165) but ~100× worse on MMD
(0.062 vs 0.00034) — right region, wrong distribution. On bimodal, MMD ≈ 0.11 for
*every* method including W1W2+GPA, so that column does not discriminate.

### 4.2 Low-dim sweep (`14_sweep/lam*/summary.json`)

Best flow+GPA: **circle λ=0.5, L=100 → 0.0107**; **bimodal λ=0.25, L=1000 → 0.0182**.

Treat the L=1000 column with care — the `GPA_only` control diverges there
(bimodal: 0.249 → 0.053 → **2.397** across L=10/100/1000; circle: 0.024 → 0.018 →
0.098). L=1000 also uses a different η (0.005 vs 0.01), so it is not a clean
comparison. Best result from a regime where the control stayed well-behaved:
**circle λ=0.5, L=100 → 0.0107**; **bimodal λ=0.1, L=100 → 0.0751**.

The published bimodal headline (0.0182) comes from the L=1000 regime.

### 4.3 MNIST (`results/mnist/sweep/*/metrics.json`)

Best flow: `rkl_gp_10k` — MSE 0.0801, diversity 0.161.
Best flow+GPA: `gpa_v4_rkl_L1000_2k` — MSE 0.0841, diversity 0.154.

**[reasoning] Diversity is the concern.** L=1 → diversity 0.243; L=1000 → 0.154. Lower MSE with
lower diversity on inpainting is the signature of collapse toward a conditional
mean rather than posterior sampling. For a paper about *posterior sampling*, MSE
alone is the wrong objective and will not reveal this.

`v2`, `v3`, `v4` L=1000 runs are byte-identical (same MSE/diversity to 5 dp) —
re-runs of one config, not a progression.

**The one matched LT vs DV comparison** (`archive/gpa_v2_kl_1k` vs
`archive/gpa_v2_rkl_1k`, same checkpoint, K=1000, η=0.5, L=1.0):

| | `kl` (LT) | `reverse_kl` (DV) |
|---|---|---|
| MSE | 0.1091 | 0.1138 |
| diversity | 0.2402 | 0.2426 |
| wall clock | **8342 s** | **147 s** |

Equivalent quality, 57× the cost. **Confounded**: these two also differ in
`lip_mode` (`project` vs `gp`), so the f\*-representation and Γ-domain questions
are entangled. A clean re-run holding `lip_mode` fixed would isolate it.

---

## 5. Open choices

Separated by kind. §5.A are defects with no judgement call — the current behaviour
does not match any stated intent. §5.B are genuine choices where I have no
evidence favouring one option.

### 5.A Defects

| # | item | ref |
|---|---|---|
| A1 | `flows/` GP ignores `lip_scale`, hardcodes threshold 1 | §2.2 |
| A2 | Current GP omits y ⇒ searches a class larger than Γ | §2.1 |
| A3 | CNF-MLE gradient path severed ⇒ `NaN` baseline | §3.1 |
| A4 | `reverse_kl` misnames the DV form of forward KL | §2.4 |
| A5 | Deleted drivers ⇒ results not reproducible from tree | §1 |
| A6 | CLI/library defaults contradict (`L`, `eta`) | §3.3 |

### 5.B Choices

**B1. How to enforce Γ.** Joint GP (`‖∇_{(x,y)}φ‖`, soft, penalty slack) vs
spectral projection (hard, `L^(1/D)` conservative). Both target Γ. I have no
evidence which suits this problem; the one empirical comparison in the tree is
confounded (§4.3).

**B2. Where to evaluate a joint GP.** At data `(x,y)` pairs, or with interpolation
extended into y. WGAN-GP's interpolation argument concerns transport in x and has
no stated y-analogue. **[reasoning]** — I have not tested whether y-interpolation
leaves `π_Y`'s support, and for the Euler problem I do not know how it behaves.

**B3. Whether to consolidate the `mnist/` fork** or keep it separate and accept
drift. §1.

**B4. MNIST reporting metric.** MSE alone does not distinguish posterior sampling
from conditional-mean collapse; the L=1→1000 diversity drop (§4.3) is consistent
with the latter but I have not confirmed the mechanism.

**B5. `µ₀` for Euler.** N(0,I) will not coincide with a
SO(3) × inverse-Wishart prior; `--source-samples` exists but is untested at that
scale. §2.8.

### 5.C Cheapest discriminating experiment

LT vs DV at **fixed** `lip_mode`, same checkpoint, same K/η/L. The existing pair
(`gpa_v2_kl_1k` / `gpa_v2_rkl_1k`) varies both, so it cannot separate the
f\*-representation question from the Γ question. One GPU-hour.
