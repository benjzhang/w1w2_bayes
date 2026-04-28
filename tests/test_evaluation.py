"""Tests for evaluation utilities."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from problems import get_problem
from flows import W1W2Flow
from utils.evaluation import generate_posterior, evaluate_flow, plot_results


class TestGeneratePosterior:
    @pytest.fixture
    def trained_flow(self):
        flow = W1W2Flow(
            theta_dim=2, y_dim=1,
            vel_hidden=32, vel_layers=2,
            device=torch.device('cpu')
        )
        # Minimal training
        theta = torch.randn(50, 2)
        y = torch.randn(50, 1)
        flow.train(theta, y, n_iters=1, batch_size=50, n_steps=3, verbose=False)
        return flow

    def test_output_shape(self, trained_flow):
        samples = generate_posterior(
            trained_flow.vel_net,
            y_obs=0.5,
            n_samples=100,
            n_steps=5,
            theta_dim=2,
            y_dim=1
        )
        assert samples.shape == (100, 2)
        assert isinstance(samples, np.ndarray)

    def test_deterministic_with_seed(self, trained_flow):
        torch.manual_seed(123)
        samples1 = generate_posterior(
            trained_flow.vel_net, y_obs=0.0,
            n_samples=20, n_steps=5, theta_dim=2, y_dim=1
        )

        torch.manual_seed(123)
        samples2 = generate_posterior(
            trained_flow.vel_net, y_obs=0.0,
            n_samples=20, n_steps=5, theta_dim=2, y_dim=1
        )

        np.testing.assert_array_equal(samples1, samples2)


class TestEvaluateFlow:
    @pytest.fixture
    def setup(self):
        problem = get_problem('linear')
        flow = W1W2Flow(
            theta_dim=2, y_dim=1,
            vel_hidden=32, vel_layers=2,
            device=torch.device('cpu')
        )
        theta = torch.randn(50, 2)
        y = torch.randn(50, 1)
        flow.train(theta, y, n_iters=1, batch_size=50, n_steps=3, verbose=False)
        return flow, problem

    def test_returns_dict_with_expected_keys(self, setup):
        flow, problem = setup
        results = evaluate_flow(
            flow.vel_net, problem,
            n_samples=50, n_steps=3
        )

        assert 'y_values' in results
        assert 'mean_distances' in results
        assert 'std_distances' in results
        assert 'samples' in results

    def test_uses_problem_default_y_values(self, setup):
        flow, problem = setup
        results = evaluate_flow(flow.vel_net, problem, n_samples=50, n_steps=3)

        expected_y = problem.default_y_test_values()
        assert results['y_values'] == expected_y

    def test_custom_y_values(self, setup):
        flow, problem = setup
        custom_y = [0.0, 0.5]
        results = evaluate_flow(
            flow.vel_net, problem,
            y_test_values=custom_y,
            n_samples=50, n_steps=3
        )

        assert results['y_values'] == custom_y
        assert len(results['mean_distances']) == 2


class TestPlotResults:
    def test_creates_figure(self):
        problem = get_problem('linear')
        flow = W1W2Flow(
            theta_dim=2, y_dim=1,
            vel_hidden=32, vel_layers=2,
            device=torch.device('cpu')
        )
        theta = torch.randn(50, 2)
        y = torch.randn(50, 1)
        history = flow.train(theta, y, n_iters=1, batch_size=50, n_steps=3, verbose=False)

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig = plot_results(
            flow.vel_net, problem, history,
            n_samples=50, n_steps=3
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_to_file(self):
        problem = get_problem('linear')
        flow = W1W2Flow(
            theta_dim=2, y_dim=1,
            vel_hidden=32, vel_layers=2,
            device=torch.device('cpu')
        )
        theta = torch.randn(50, 2)
        y = torch.randn(50, 1)
        history = flow.train(theta, y, n_iters=1, batch_size=50, n_steps=3, verbose=False)

        import matplotlib
        matplotlib.use('Agg')

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            fig = plot_results(
                flow.vel_net, problem, history,
                save_path=str(save_path),
                n_samples=50, n_steps=3
            )

            assert save_path.exists()

            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_with_run_config(self):
        problem = get_problem('linear')
        flow = W1W2Flow(
            theta_dim=2, y_dim=1,
            vel_hidden=32, vel_layers=2,
            device=torch.device('cpu')
        )
        theta = torch.randn(50, 2)
        y = torch.randn(50, 1)
        history = flow.train(theta, y, n_iters=1, batch_size=50, n_steps=3, verbose=False)

        import matplotlib
        matplotlib.use('Agg')

        run_config = {
            'run_id': 'test_run',
            'epochs': 1,
            'lam': 0.01,
            'lip_scale': 10.0,
            'vel_layers': 2,
            'vel_hidden': 32,
            'n_steps': 3,
        }

        fig = plot_results(
            flow.vel_net, problem, history,
            run_config=run_config,
            n_samples=50, n_steps=3
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
