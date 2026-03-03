"""Tests for flow implementations."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from flows import W1W2Flow
from problems import get_problem


class TestW1W2Flow:
    @pytest.fixture
    def flow(self):
        return W1W2Flow(
            theta_dim=2,
            y_dim=1,
            vel_hidden=32,
            vel_layers=2,
            disc_hidden=32,
            lip_scale=1.0,
            device=torch.device('cpu')
        )

    def test_dimensions(self, flow):
        assert flow.theta_dim == 2
        assert flow.y_dim == 1

    def test_sample_shape(self, flow):
        y = torch.tensor([1.0])
        samples = flow.sample(y, n_samples=100, n_steps=5)
        assert samples.shape == (100, 2)

    def test_sample_scalar_y(self, flow):
        y = torch.tensor(1.0)
        samples = flow.sample(y, n_samples=50, n_steps=5)
        assert samples.shape == (50, 2)

    def test_train_returns_history(self, flow):
        # Minimal training
        theta = torch.randn(100, 2)
        y = torch.randn(100, 1)

        history = flow.train(
            theta, y,
            n_epochs=2,
            batch_size=50,
            n_steps=5,
            disc_updates=1,
            verbose=False
        )

        assert 'L_dual' in history
        assert 'KE' in history
        assert len(history['L_dual']) == 2
        assert len(history['KE']) == 2

    def test_save_load_roundtrip(self, flow):
        # Train briefly to change weights
        theta = torch.randn(50, 2)
        y = torch.randn(50, 1)
        flow.train(theta, y, n_epochs=1, batch_size=50, n_steps=3, verbose=False)

        # Sample before save
        y_test = torch.tensor([0.5])
        torch.manual_seed(42)
        samples_before = flow.sample(y_test, n_samples=10, n_steps=5)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            flow.save(str(path))

            # Create new flow and load
            flow2 = W1W2Flow(
                theta_dim=2, y_dim=1,
                vel_hidden=32, vel_layers=2,
                disc_hidden=32, lip_scale=1.0,
                device=torch.device('cpu')
            )
            flow2.load(str(path))

            # Sample after load
            torch.manual_seed(42)
            samples_after = flow2.sample(y_test, n_samples=10, n_steps=5)

            torch.testing.assert_close(samples_before, samples_after)

    def test_from_checkpoint(self, flow):
        theta = torch.randn(50, 2)
        y = torch.randn(50, 1)
        flow.train(theta, y, n_epochs=1, batch_size=50, n_steps=3, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            flow.save(str(path))

            flow2 = W1W2Flow.from_checkpoint(str(path), device=torch.device('cpu'))

            assert flow2.theta_dim == flow.theta_dim
            assert flow2.y_dim == flow.y_dim

    def test_checkpoint_during_training(self):
        flow = W1W2Flow(
            theta_dim=2, y_dim=1,
            vel_hidden=32, vel_layers=2,
            disc_hidden=32,
            device=torch.device('cpu')
        )

        theta = torch.randn(100, 2)
        y = torch.randn(100, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            flow.train(
                theta, y,
                n_epochs=3,
                batch_size=50,
                n_steps=3,
                checkpoint_dir=tmpdir,
                checkpoint_every=2,
                verbose=False
            )

            # Should have checkpoint at epoch 2
            checkpoint_path = Path(tmpdir) / "checkpoint_epoch2.pt"
            assert checkpoint_path.exists()


class TestW1W2FlowWithProblem:
    """Integration tests with actual problems."""

    def test_linear_problem_shapes(self):
        problem = get_problem('linear')
        flow = W1W2Flow(
            theta_dim=problem.theta_dim,
            y_dim=problem.y_dim,
            vel_hidden=32,
            vel_layers=2,
            device=torch.device('cpu')
        )

        theta_np, y_np = problem.sample_joint(100)
        theta = torch.FloatTensor(theta_np)
        y = torch.FloatTensor(y_np).unsqueeze(1)

        history = flow.train(theta, y, n_epochs=1, batch_size=50, n_steps=3, verbose=False)
        assert len(history['L_dual']) == 1

        samples = flow.sample(torch.tensor([0.0]), n_samples=50, n_steps=3)
        assert samples.shape == (50, 2)

    def test_quadratic_problem_with_quad_features(self):
        problem = get_problem('quadratic')
        flow = W1W2Flow(
            theta_dim=problem.theta_dim,
            y_dim=problem.y_dim,
            vel_hidden=32,
            vel_layers=2,
            use_quadratic_features=True,
            device=torch.device('cpu')
        )

        theta_np, y_np = problem.sample_joint(100)
        theta = torch.FloatTensor(theta_np)
        y = torch.FloatTensor(y_np).unsqueeze(1)

        history = flow.train(theta, y, n_epochs=1, batch_size=50, n_steps=3, verbose=False)
        assert len(history['L_dual']) == 1
