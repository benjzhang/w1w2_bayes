"""FitzHugh-Nagumo SDE inference problem.

SDE:
    dv = (v - v^3/3 - w + I) dt + sigma dW
    dw = epsilon * (v + a - b*w) dt

Parameters: theta = (a, I, w0, log b, log eps, log sigma)
Observations: y_n = v(t_n) + eta_n, eta_n ~ N(0, sigma_obs^2)

The transformation (a, I, w0) -> (a+b*delta, I+delta, w0+delta) leaves v(t)
pathwise invariant, so the posterior concentrates on a 5D manifold in 6D.
"""

import numpy as np
from typing import Tuple, List, Optional
from .base import BaseProblem


class FitzHughNagumoProblem(BaseProblem):
    """FitzHugh-Nagumo SDE parameter inference.

    Args:
        T: Time horizon.
        N_obs: Number of observation times (evenly spaced in (0, T]).
        dt: Euler-Maruyama step size.
        sigma_obs: Observation noise standard deviation.
        v0: Known initial voltage.
        prior_mu: Prior means for [a, I, w0, log(b), log(eps), log(sigma)].
        prior_sigma: Prior stds for the same.
    """

    def __init__(
        self,
        T: float = 10.0,
        N_obs: int = 20,
        dt: float = 0.005,
        sigma_obs: float = 0.5,
        v0: float = -1.0,
        prior_mu: Optional[List[float]] = None,
        prior_sigma: Optional[List[float]] = None,
    ):
        self.T = T
        self.N_obs = N_obs
        self.dt = dt
        self.sigma_obs = sigma_obs
        self.v0 = v0

        self.obs_times = np.linspace(T / N_obs, T, N_obs)
        self.obs_indices = np.round(self.obs_times / dt).astype(int)

        if prior_mu is None:
            prior_mu = [0.7, 0.3, 0.0, np.log(0.8), np.log(0.08), np.log(0.3)]
        if prior_sigma is None:
            prior_sigma = [0.5, 0.5, 0.5, 0.3, 0.5, 0.5]
        self.prior_mu = np.array(prior_mu)
        self.prior_sigma = np.array(prior_sigma)

        # True parameters for generating test observations
        self.theta_true = np.array([0.7, 0.3, -1.0,
                                    np.log(0.8), np.log(0.08), np.log(0.3)])

    @property
    def name(self) -> str:
        return "fitzhugh_nagumo"

    @property
    def theta_dim(self) -> int:
        return 6

    @property
    def y_dim(self) -> int:
        return self.N_obs

    @property
    def description(self) -> str:
        return (f"FitzHugh-Nagumo SDE: 6 params, {self.N_obs} observations, "
                f"T={self.T}, sigma_obs={self.sigma_obs}")

    def _simulate_sde_batch(self, theta: np.ndarray,
                            seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized Euler-Maruyama for the FHN SDE.

        Args:
            theta: (n, 6) — [a, I, w0, log_b, log_eps, log_sigma]
            seed: Optional random seed for reproducibility.

        Returns:
            v_obs: (n, N_obs) — v at observation times (no observation noise)
            blowup: (n,) bool — True if trajectory diverged
        """
        if seed is not None:
            np.random.seed(seed)

        n = len(theta)
        a = theta[:, 0]
        I = theta[:, 1]
        w0 = theta[:, 2]
        b = np.exp(theta[:, 3])
        eps = np.exp(theta[:, 4])
        sigma = np.exp(theta[:, 5])

        n_steps = int(self.T / self.dt)
        sqrt_dt = np.sqrt(self.dt)

        v = np.full(n, self.v0)
        w = w0.copy()
        v_obs = np.zeros((n, self.N_obs))
        obs_idx = 0

        for step in range(1, n_steps + 1):
            dW = np.random.randn(n) * sqrt_dt
            v = v + (v - v**3 / 3 - w + I) * self.dt + sigma * dW
            w = w + eps * (v + a - b * w) * self.dt
            v = np.clip(v, -100, 100)
            w = np.clip(w, -100, 100)

            if obs_idx < self.N_obs and step == self.obs_indices[obs_idx]:
                v_obs[:, obs_idx] = v
                obs_idx += 1

        blowup = np.any(np.abs(v_obs) > 20, axis=1)
        return v_obs, blowup

    def sample_joint(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (theta, y) from the joint distribution.

        Draws theta from prior, simulates SDE, adds observation noise.
        Rejects and resamples trajectories that blow up.
        """
        collected_theta = []
        collected_y = []
        n_collected = 0
        oversample = 1.3

        while n_collected < n:
            n_try = int((n - n_collected) * oversample) + 100
            theta = (np.random.randn(n_try, 6) * self.prior_sigma[None, :]
                     + self.prior_mu[None, :])
            v_obs, blowup = self._simulate_sde_batch(theta)
            valid = ~blowup
            if valid.sum() == 0:
                oversample *= 2
                continue
            y = v_obs[valid] + self.sigma_obs * np.random.randn(valid.sum(), self.N_obs)
            collected_theta.append(theta[valid])
            collected_y.append(y)
            n_collected += valid.sum()

        theta_out = np.concatenate(collected_theta, axis=0)[:n]
        y_out = np.concatenate(collected_y, axis=0)[:n]
        return theta_out, y_out

    def generate_true_observation(self, seed: int = 123) -> np.ndarray:
        """Generate a single observation vector from the true parameters."""
        v_obs, _ = self._simulate_sde_batch(self.theta_true[None, :], seed=seed)
        rng = np.random.RandomState(seed)
        y = v_obs[0] + self.sigma_obs * rng.randn(self.N_obs)
        return y

    def compute_distance(self, theta: np.ndarray, y_obs: float) -> np.ndarray:
        """Not implemented — no closed-form manifold distance for SDE problems."""
        raise NotImplementedError(
            "compute_distance not available for FHN. Use posterior predictive checks instead.")

    def sample_true_posterior(self, y_obs, n_samples: int) -> np.ndarray:
        """Not available — no closed-form posterior for SDE problems."""
        raise NotImplementedError(
            "No closed-form posterior for FHN. Use ABC or MCMC for reference.")

    def default_y_test_values(self) -> List[float]:
        """Return true observation as default test value."""
        return [self.generate_true_observation()]

    def default_hyperparams(self) -> dict:
        return {
            'n_iters': 15000,
            'batch_size': 512,
            'lr': 5e-4,
            'lam': 0.005,
            'n_steps': 50,
            'disc_updates': 5,
            'lip_scale': 10.0,
            'gp_lambda': 1.0,
            'vel_hidden': 512,
            'vel_layers': 5,
        }
