# Notes from Ben — context for the audit

Statements from the author about intent and history. These take precedence over
inferences in `AUDIT.md`. Recorded as given; **not yet processed into the audit.**

---

## On §1 (three parallel implementations)

**Multiple implementations are not automatically drift.** Some are alternate
mathematical implementations, all valid depending on implementation choices.
Do not assume divergence between copies means one is wrong.

**One of the MNIST implementations may be a reference from a previous
implementation** — kept for comparison, not intended to be run. (Which one is not
yet specified.)

---

## On §2.1 (Γ / Lipschitz enforcement)

**Q: is "projection" spectral normalization?** Essentially yes — but note the
codebase has *two* distinct mechanisms, and only one is in the active path:

- `_spectral_norm_projection` (`refinement/gpa.py:48`, and copies in `mnist/`):
  **hard rescale after each optimizer step**, `W ← L^(1/D) · W/‖W‖₂`. This is what
  `lip_mode='project'` uses. Spectral normalization in spirit, applied as a
  projection step rather than a reparametrization.
- `nn.utils.spectral_norm` (PyTorch's reparametrizing wrapper) appears only in
  `nn/discriminator.py:66` — and **`nn/Discriminator` is not used by the active
  path**; `W1W2Flow` constructs a `GPADiscriminator` instead. `gpa.py:176` only
  calls `remove_spectral_norm` defensively.

So "projection" = hard rescale, not the PyTorch wrapper.

**Author's experience:** spectral-normalized / projected nets seemed to **lack
expressivity**. Gradient penalty generally **performed better**.

**Intended direction:** for consistency with the math, add an implementation that
penalizes the y variable as well. **Put a flag** on it so the code can switch
between x-only and joint (x,y) penalization.

⇒ This resolves §5.B1: not a choice between GP and projection — keep GP as the
preferred mechanism, and make the Γ-domain a switchable flag. §5.A2 becomes
"add the joint option", not "replace GP".

---

## On §2.2 (`flows/` GP hardcodes threshold at 1)

**Confirmed an error by author.** The threshold should be customizable like `L`,
not fixed at 1.

Current: `w1w2_flow.py:143` → `mean(max(0, ‖∇φ‖² − 1))`, ignores `lip_scale`.

Note when fixing: `flows/` and `mnist/` use `‖g‖² − L²` form; `refinement/gpa.py`
uses `max(0, ‖g‖ − L)²`. Author has said (§2.3) differing GP *forms* are
acceptable, so this fix is about making the **threshold** a parameter, not about
unifying the formulas.

Consequence for existing results: every GP-mode run — including all of
`14_sweep` (`flow_gp_lambda=1.0`) — was constrained to Γ₁ regardless of
`lip_scale=10`. Affects interpretation of the published baseline numbers.

---

## On §2.3 (three GP formulas)

**Author: the three forms seem weird but are valid.** Not to be unified. The
requirement is only that the threshold be changeable, never hardcoded to 1.

Status of the three, re hardcoding:

| site | formula | threshold |
|---|---|---|
| `flows/w1w2_flow.py:143` | `mean(max(0, ‖g‖² − 1))` | **hardcoded 1 — needs fix** |
| `refinement/gpa.py:249` | `mean(max(0, ‖g‖ − L)²)` | already uses `L` — OK |
| `mnist/run_one.py` | `mean(max(0, ‖g‖² − L²))` | already uses `L` — OK |

⇒ Only `flows/w1w2_flow.py` needs changing. This is the same fix as §2.2 (they
are one defect, not two — §2.2 *is* the hardcoded-1 in the `flows/` GP).

Note the two remaining forms differ in shape, not just threshold: `‖g‖²−L²`
penalizes quadratically in ‖g‖², `max(0,‖g‖−L)²` quadratically in the excess.
Author considers both valid.

---

## On §2.4 (`fstar` naming / LT vs DV)

**Author:** Donsker–Varadhan is valid, and the dual of `f(t) = t log t` is valid —
both good. **Different applications may produce different results**, so keeping
both is intentional, not redundancy. The `LT_nu` (extra optimized parameter)
variant exists but is **out of scope for now**.

**Instability of raw KL at large L is expected** — a natural property of the
method, not a bug. Tuning keeps it away. ⇒ Retract the framing in AUDIT §2.4
that treated LT overflow as a defect; it is a known property. The *silent*
`NaN`-write (§3.2) remains a separate, real issue.

**Verified (exhaustive grep over all non-attic `.py`): no reverse KL anywhere.**
Every dual site is one of:
- `phi.mean() - exp(phi_real - 1).mean()` — LT, dual of `f(t)=t log t`
- `phi.mean() - logsumexp(phi_real) + log(n)` — DV

Sites: `flows/w1w2_flow.py:230,251`; `refinement/gpa.py:246,249,252`;
`mnist/run_one.py:153,156,343,345`; `mnist/run_ref_gpa.py:218,220`;
`mnist/train_flow.py:135,152`; `mnist/gpa_refine.py:160`;
`mnist/sweep.py:206,218,342`. (attic/ likewise, all LT.)

True reverse KL would need `f(t) = −log t`, `f*(s) = −1 − log(−s)`, `s<0` — that
form appears **nowhere**. So the author's recollection is right: the method never
used reverse KL. Only the **label** `--fstar reverse_kl` in `mnist/` is wrong; it
selects DV. Renaming is cosmetic and changes no result.

---

## On §2.5 (`f*` hardcoded to KL in `flows/`)

**Author:** the Lipschitz-regularized KL implementation is probably fine.
Hardcoding is not great long-term but acceptable for now. Decision delegated.

**My call: make it flexible, bundled with the §2.2 fix — not as its own task.**

Reasoning: `f*` is *already* switchable in `mnist/` (`--fstar`) and
`refinement/gpa.py` (`formulation`); `flows/` is the lone place it is fixed. That
asymmetry blocks running LT vs DV on the low-dim problems without editing source
— and that is precisely the clean comparison currently missing (the only existing
LT/DV pair is confounded with `lip_mode`, §4.3). So the justification is
unblocking a wanted experiment, not tidiness.

Scope limit: add an `fstar` argument accepting the **two forms already
implemented** (LT, DV), defaulting to LT so existing behaviour is unchanged.
Do **not** build a general `f`-conjugate registry — `LT_nu` is out of scope and
nothing in the paper's experiments needs a third form.

Both changes land in `flows/w1w2_flow.py`, same pass as §2.2.

---

## On §2.6 (λ vs ½ placement)

**Author:** λ separate from ½ is fine, but must be **consistent across all
implementations** so runs are comparable.

**Verified: already consistent.** Every non-attic site uses the identical
convention — ½ inside the KE accumulation, λ multiplying the whole thing:

```
KE += 0.5 * (v**2).sum(dim=1).mean() * dt
loss = L_dual + lam * KE
```

| site | KE form | λ use |
|---|---|---|
| `flows/w1w2_flow.py:259,261` | `0.5*…*dt` | `lam * KE` |
| `mnist/run_one.py:195,197` | same | `args.lam * KE` |
| `mnist/sweep.py:225,227` | same | `cfg.lam * KE` |
| `mnist/train_flow.py:160,162` | same | `args.lam * KE` |
| `utils/integrators.py:63` | same | (helper, unused) |
| `baselines/cnf.py:146,149` | same | `lam * ke` |

⇒ **No action needed.** λ values are directly comparable across every path,
including the CNF baseline. (One attic file, `conditional_ot_flow.py:183`, uses
`(v**2).mean()` without `.sum(dim=1)` — divides by dim — but attic is out of
scope.)

Caveat for comparability: `baselines/cnf.py` applies λ to a **different** first
term (`nll`, not `L_dual`), so `cnf_lam` is not on the same scale as `flow_lam`
even though the KE convention matches. The `14_sweep` baselines used
`flow_lam=0.25`, `cnf_lam=0.01`.

---

## On §2.7 (spectral projection as surrogate)

**Author:** projection is not a Lipschitz constraint; it is a fine **surrogate**.
The point is only that the implementation be correct. ⇒ Retract AUDIT §2.7's
framing of `L^(1/D)` looseness as a deficiency.

Checked the implementation. **Two findings.**

**[verified] The arithmetic is right.** `_project_disc_weights` sets each layer to
`L^(1/D)`; product = `L` exactly (tested L=1 → 1.0000, L=10 → 10.0000, D=4).

**[verified] Surrogate validity requires 1-Lipschitz activations. Measured:**

| activation | max │f′│ | |
|---|---|---|
| ReLU | 1.0000 | OK |
| MollifiedReLU(0.5) | 1.0000 | OK |
| **SiLU** | **1.0998** | **exceeds 1** |

Default pairing is correct: `gp_lambda==0 → mollified_relu + projection`;
`gp_lambda>0 → silu + no projection` (`w1w2_flow.py:88`).

**⚠ [verified] The warm-start path breaks that pairing.** `run_e2e_sweep.py`
passes `disc=flow.disc` into `gpa_refine`. With `flow_gp_lambda=1.0` the flow disc
is **SiLU**; `gpa_refine` with `gp_weight==0` then **projects** it
(`gpa.py:190-191`). So projection is applied to a SiLU net whose activations are
1.0998-Lipschitz. The `activation='mollified_relu'` argument is ignored on the
warm-start branch — it only applies when `disc is None`.

This is exactly the `14_sweep` configuration (`flow_gp_lambda=1.0`,
`gp_weight=0.0`, `activation=mollified_relu`), i.e. **every published
flow+GPA number**. Effect: realized Lipschitz bound is `L · 1.0998^(D-1)`
(≈1.33× for D=4), not `L`. Small, but the surrogate is not computing what it
claims. Also relates to §3.4 (warm-start inherits weights but not regime).

**Suggested fix (not applied):** on the warm-start branch, either rebuild the disc
with the requested activation, or refuse to project a net whose activations are
not 1-Lipschitz.

---

## On §2.8 (µ₀ = prior)

**Author: the source should generally equal the prior.** If the Euler problem
does not do this, change it accordingly.

**[verified] The Euler problem does not exist yet.** `problems/` holds only
`base, circle, bimodal_quadratic, quadratic, linear, fitzhugh_nagumo`. No
SO(3)/inertia/Wishart code anywhere. Paper §3 is spec, not implementation — so
nothing to change there yet; it is a requirement for when it *is* written.

**⚠ [verified] But FitzHugh–Nagumo already has the mismatch.** Its prior is
Gaussian with **non-unit mean and scale**:

```
prior_mu    = [0.7, 0.3, 0.0, log(0.8), log(0.08), log(0.3)]
prior_sigma = [0.5, 0.5, 0.5, 0.3, 0.5, 0.5]
theta = randn(n,6) * prior_sigma + prior_mu
```

so `π_pr ≠ N(0,I)`. `W1W2Flow.train` defaults `z = randn(...)` when
`source_samples is None`, i.e. **µ₀ = N(0,I) ≠ π_pr** for FHN.

**[verified] Nothing ever passes `source_samples`.** The plumbing exists
(`pipeline.py:428` `--source-samples`; `w1w2_flow.py:217-219`, `:319-320`) but
no driver, launcher, or sweep supplies it. `run_e2e_sweep.py` uses plain
`torch.randn` for both the flow source and the "GPA-only from prior" control.

⇒ Consequences:
- circle / bimodal_quadratic / quadratic / linear: prior **is** N(0,I), so no
  issue (§2.8 of AUDIT stands).
- **fitzhugh_nagumo: µ₀ ≠ prior.** If FHN has been run, µ₀ was wrong. Also the
  "GPA-only from prior" control would be misnamed for FHN.
- Euler: must supply `source_samples` from the SO(3) × inverse-Wishart prior when
  implemented.

**Open question for author:** has FHN actually been run? No FHN results exist in
`results/`, so possibly never exercised.

---

*(awaiting further points — §3 onward)*
