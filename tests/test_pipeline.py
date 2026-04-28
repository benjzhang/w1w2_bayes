"""End-to-end tests for the unified pipeline.

Tests both raw data (.npz) and named problem inputs.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def synthetic_data_path(tmp_dir):
    """Create a simple synthetic joint dataset: y = theta_0 + noise."""
    np.random.seed(42)
    n = 300
    theta = np.random.randn(n, 3).astype(np.float32)
    y = (theta[:, 0:1] + 0.1 * np.random.randn(n, 1)).astype(np.float32)
    path = tmp_dir / 'data.npz'
    np.savez(path, theta=theta, y=y)
    return path


# Small hyperparams for fast tests
FAST_TRAIN = dict(n_iters=20, batch_size=64, lr=1e-3, lam=0.01, n_steps=5,
                  disc_updates=1, vel_hidden=32, vel_layers=2,
                  disc_hidden=16, disc_layers=2, lip_scale=10.0,
                  gp_lambda=0.0, checkpoint_every=0)

FAST_GPA = dict(K=5, eta=0.01, L=1.0, disc_steps=1, disc_lr=0.001,
                batch_size=64, gp_weight=0.0, disc_hidden=16,
                disc_layers=2, formulation='LT')


# ── Step 1: Train ──

class TestTrain:
    def test_train_from_npz(self, tmp_dir, synthetic_data_path):
        from pipeline import cmd_train
        args = _make_args('train', data=str(synthetic_data_path),
                          output=str(tmp_dir / 'run'), seed=42, **FAST_TRAIN)
        cmd_train(args)
        assert (tmp_dir / 'run' / 'model.pt').exists()
        assert (tmp_dir / 'run' / 'hparams.json').exists()

    def test_train_from_problem(self, tmp_dir):
        from pipeline import cmd_train
        args = _make_args('train', problem='circle', n_train=200,
                          output=str(tmp_dir / 'run'), seed=42, **FAST_TRAIN)
        cmd_train(args)
        assert (tmp_dir / 'run' / 'model.pt').exists()

    def test_train_reloads(self, tmp_dir, synthetic_data_path):
        from pipeline import cmd_train
        from flows import W1W2Flow
        args = _make_args('train', data=str(synthetic_data_path),
                          output=str(tmp_dir / 'run'), seed=42, **FAST_TRAIN)
        cmd_train(args)
        flow = W1W2Flow.from_checkpoint(str(tmp_dir / 'run' / 'model.pt'))
        assert flow.theta_dim == 3
        assert flow.y_dim == 1


# ── Step 2: Sample ──

class TestSample:
    def test_sample_basic(self, tmp_dir, synthetic_data_path):
        _train(tmp_dir, synthetic_data_path)
        from pipeline import cmd_sample
        args = _make_args('sample',
                          model=str(tmp_dir / 'run' / 'model.pt'),
                          y_values=[0.0, 1.0], n_samples=50, n_steps=5,
                          output=str(tmp_dir / 'samples.npz'), seed=42)
        cmd_sample(args)

        s = np.load(tmp_dir / 'samples.npz')
        assert s['theta'].shape == (100, 3)  # 2 obs × 50 samples
        assert s['y'].shape == (100, 1)
        assert s['y_obs'].shape == (2, 1)
        assert int(s['n_samples_per_obs']) == 50

    def test_sample_multidim_y(self, tmp_dir):
        """Test with y_dim=2 using a raw data file."""
        np.random.seed(42)
        theta = np.random.randn(200, 2).astype(np.float32)
        y = np.column_stack([theta[:, 0]**2, theta[:, 1]]).astype(np.float32)
        data_path = tmp_dir / 'data_2d.npz'
        np.savez(data_path, theta=theta, y=y)

        from pipeline import cmd_train, cmd_sample
        args = _make_args('train', data=str(data_path),
                          output=str(tmp_dir / 'run2d'), seed=42, **FAST_TRAIN)
        cmd_train(args)

        # y_dim=2, so 4 values = 2 observations
        args = _make_args('sample',
                          model=str(tmp_dir / 'run2d' / 'model.pt'),
                          y_values=[0.5, 1.0, 1.5, 2.0],
                          n_samples=30, n_steps=5,
                          output=str(tmp_dir / 'samp2d.npz'), seed=42)
        cmd_sample(args)
        s = np.load(tmp_dir / 'samp2d.npz')
        assert s['theta'].shape == (60, 2)  # 2 obs × 30 samples
        assert s['y_obs'].shape == (2, 2)


# ── Step 3: Refine ──

class TestRefine:
    def test_refine_from_npz(self, tmp_dir, synthetic_data_path):
        _train(tmp_dir, synthetic_data_path)
        _sample(tmp_dir)

        from pipeline import cmd_refine
        args = _make_args('refine',
                          samples=str(tmp_dir / 'samples.npz'),
                          data=str(synthetic_data_path),
                          model=str(tmp_dir / 'run' / 'model.pt'),
                          n_steps=5,
                          output=str(tmp_dir / 'refined.npz'),
                          seed=42, **FAST_GPA)
        cmd_refine(args)

        r = np.load(tmp_dir / 'refined.npz')
        assert r['theta'].shape[0] == 100  # same as input samples
        assert r['theta'].shape[1] == 3

    def test_refine_from_problem(self, tmp_dir):
        from pipeline import cmd_train, cmd_sample, cmd_refine
        args = _make_args('train', problem='circle', n_train=200,
                          output=str(tmp_dir / 'run'), seed=42, **FAST_TRAIN)
        cmd_train(args)

        args = _make_args('sample',
                          model=str(tmp_dir / 'run' / 'model.pt'),
                          y_values=[1.0], n_samples=50, n_steps=5,
                          output=str(tmp_dir / 'samples.npz'), seed=42)
        cmd_sample(args)

        args = _make_args('refine',
                          samples=str(tmp_dir / 'samples.npz'),
                          problem='circle', n_train=200,
                          model=str(tmp_dir / 'run' / 'model.pt'),
                          n_steps=5,
                          output=str(tmp_dir / 'refined.npz'),
                          seed=42, **FAST_GPA)
        cmd_refine(args)
        r = np.load(tmp_dir / 'refined.npz')
        assert r['theta'].shape == (50, 2)

    def test_refine_without_model(self, tmp_dir, synthetic_data_path):
        """Refine without coupled flow particles (no --model)."""
        _train(tmp_dir, synthetic_data_path)
        _sample(tmp_dir)

        from pipeline import cmd_refine
        args = _make_args('refine',
                          samples=str(tmp_dir / 'samples.npz'),
                          data=str(synthetic_data_path),
                          model=None, n_steps=5,
                          output=str(tmp_dir / 'refined.npz'),
                          seed=42, **FAST_GPA)
        cmd_refine(args)
        r = np.load(tmp_dir / 'refined.npz')
        assert r['theta'].shape == (100, 3)


# ── Step 4: Plot ──

class TestPlot:
    def test_corner_plot_2d(self, tmp_dir, synthetic_data_path):
        _train(tmp_dir, synthetic_data_path)
        _sample(tmp_dir)

        from pipeline import cmd_plot
        args = _make_args('plot',
                          samples=str(tmp_dir / 'samples.npz'),
                          refined=None, true_data=None, problem=None,
                          output=str(tmp_dir / 'plots'),
                          labels=None, max_points=500)
        cmd_plot(args)
        # Should produce 2 plots (2 y observations)
        pngs = list((tmp_dir / 'plots').glob('corner_*.png'))
        assert len(pngs) == 2

    def test_corner_plot_with_problem_overlay(self, tmp_dir):
        from pipeline import cmd_train, cmd_sample, cmd_plot
        args = _make_args('train', problem='circle', n_train=200,
                          output=str(tmp_dir / 'run'), seed=42, **FAST_TRAIN)
        cmd_train(args)
        args = _make_args('sample',
                          model=str(tmp_dir / 'run' / 'model.pt'),
                          y_values=[1.0], n_samples=100, n_steps=5,
                          output=str(tmp_dir / 'samples.npz'), seed=42)
        cmd_sample(args)
        args = _make_args('plot',
                          samples=str(tmp_dir / 'samples.npz'),
                          refined=None, true_data=None, problem='circle',
                          output=str(tmp_dir / 'plots'),
                          labels=None, max_points=500)
        cmd_plot(args)
        pngs = list((tmp_dir / 'plots').glob('corner_*.png'))
        assert len(pngs) == 1

    def test_corner_plot_5d(self, tmp_dir):
        """Test corner plot scales to 5 dimensions."""
        from pipeline import make_corner_plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        np.random.seed(42)
        samp = np.random.randn(500, 5).astype(np.float32)
        fig = make_corner_plot({'test': (samp, 'blue')}, theta_dim=5)
        out = tmp_dir / 'corner_5d.png'
        fig.savefig(out, dpi=100)
        plt.close(fig)
        assert out.exists()


# ── End-to-end ──

class TestEndToEnd:
    def test_full_pipeline_npz(self, tmp_dir, synthetic_data_path):
        from pipeline import cmd_train, cmd_sample, cmd_refine, cmd_plot

        # Train
        cmd_train(_make_args('train', data=str(synthetic_data_path),
                             output=str(tmp_dir / 'run'), seed=42, **FAST_TRAIN))
        # Sample
        cmd_sample(_make_args('sample',
                              model=str(tmp_dir / 'run' / 'model.pt'),
                              y_values=[0.0], n_samples=50, n_steps=5,
                              output=str(tmp_dir / 'samples.npz'), seed=42))
        # Refine
        cmd_refine(_make_args('refine',
                              samples=str(tmp_dir / 'samples.npz'),
                              data=str(synthetic_data_path),
                              model=str(tmp_dir / 'run' / 'model.pt'),
                              n_steps=5,
                              output=str(tmp_dir / 'refined.npz'),
                              seed=42, **FAST_GPA))
        # Plot
        cmd_plot(_make_args('plot',
                            samples=str(tmp_dir / 'samples.npz'),
                            refined=str(tmp_dir / 'refined.npz'),
                            true_data=None, problem=None,
                            output=str(tmp_dir / 'plots'),
                            labels=None, max_points=200))

        assert (tmp_dir / 'run' / 'model.pt').exists()
        assert (tmp_dir / 'samples.npz').exists()
        assert (tmp_dir / 'refined.npz').exists()
        assert len(list((tmp_dir / 'plots').glob('*.png'))) > 0

    def test_full_pipeline_problem(self, tmp_dir):
        from pipeline import cmd_train, cmd_sample, cmd_refine, cmd_plot

        cmd_train(_make_args('train', problem='bimodal_quadratic', n_train=300,
                             output=str(tmp_dir / 'run'), seed=42, **FAST_TRAIN))
        cmd_sample(_make_args('sample',
                              model=str(tmp_dir / 'run' / 'model.pt'),
                              y_values=[0.0, 1.0], n_samples=100, n_steps=5,
                              output=str(tmp_dir / 'samples.npz'), seed=42))
        cmd_refine(_make_args('refine',
                              samples=str(tmp_dir / 'samples.npz'),
                              problem='bimodal_quadratic', n_train=300,
                              model=str(tmp_dir / 'run' / 'model.pt'),
                              n_steps=5,
                              output=str(tmp_dir / 'refined.npz'),
                              seed=42, **FAST_GPA))
        cmd_plot(_make_args('plot',
                            samples=str(tmp_dir / 'samples.npz'),
                            refined=str(tmp_dir / 'refined.npz'),
                            true_data=None, problem='bimodal_quadratic',
                            output=str(tmp_dir / 'plots'),
                            labels=None, max_points=200))

        r = np.load(tmp_dir / 'refined.npz')
        assert r['theta'].shape == (200, 2)
        assert len(list((tmp_dir / 'plots').glob('*.png'))) == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(command, **kwargs):
    """Build a namespace that looks like parsed CLI args."""
    from types import SimpleNamespace
    kwargs['command'] = command
    # Fill in defaults for optional fields
    kwargs.setdefault('data', None)
    kwargs.setdefault('problem', None)
    kwargs.setdefault('n_train', 10000)
    kwargs.setdefault('seed', 42)
    return SimpleNamespace(**kwargs)


def _train(tmp_dir, data_path):
    """Helper to train a small model."""
    from pipeline import cmd_train
    if not (tmp_dir / 'run' / 'model.pt').exists():
        args = _make_args('train', data=str(data_path),
                          output=str(tmp_dir / 'run'), **FAST_TRAIN)
        cmd_train(args)


def _sample(tmp_dir):
    """Helper to generate samples."""
    if not (tmp_dir / 'samples.npz').exists():
        from pipeline import cmd_sample
        args = _make_args('sample',
                          model=str(tmp_dir / 'run' / 'model.pt'),
                          y_values=[0.0, 1.0], n_samples=50, n_steps=5,
                          output=str(tmp_dir / 'samples.npz'))
        cmd_sample(args)
