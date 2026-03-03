"""Tests for inverse problem definitions."""

import pytest
import numpy as np
from problems import get_problem, PROBLEMS, LinearProblem, QuadraticProblem, CircleProblem


class TestProblemRegistry:
    def test_all_problems_registered(self):
        assert 'linear' in PROBLEMS
        assert 'quadratic' in PROBLEMS
        assert 'circle' in PROBLEMS

    def test_get_problem(self):
        for name in PROBLEMS:
            problem = get_problem(name)
            assert problem is not None
            assert problem.name == name

    def test_get_unknown_problem_raises(self):
        with pytest.raises(ValueError, match="Unknown problem"):
            get_problem("nonexistent")


class TestLinearProblem:
    @pytest.fixture
    def problem(self):
        return LinearProblem()

    def test_dimensions(self, problem):
        assert problem.theta_dim == 2
        assert problem.y_dim == 1

    def test_sample_joint_shapes(self, problem):
        n = 100
        theta, y = problem.sample_joint(n)
        assert theta.shape == (n, 2)
        assert y.shape == (n,)

    def test_sample_joint_constraint(self, problem):
        """y should equal theta_1 + theta_2."""
        theta, y = problem.sample_joint(100)
        np.testing.assert_allclose(y, theta[:, 0] + theta[:, 1])

    def test_compute_distance(self, problem):
        # Points on the line y = 1 should have distance 0
        theta_on_line = np.array([[0.5, 0.5], [1.0, 0.0], [-1.0, 2.0]])
        dist = problem.compute_distance(theta_on_line, y_obs=1.0)
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

        # Point (0, 0) has distance 1/sqrt(2) from line y=1
        theta_off_line = np.array([[0.0, 0.0]])
        dist = problem.compute_distance(theta_off_line, y_obs=1.0)
        np.testing.assert_allclose(dist, 1.0 / np.sqrt(2), atol=1e-10)

    def test_true_posterior_pdf(self, problem):
        grid = np.linspace(-3, 3, 100)
        pdf = problem.true_posterior_pdf(grid, y_obs=0.0)
        assert pdf is not None
        assert len(pdf) == len(grid)
        assert np.all(pdf >= 0)
        # Should integrate to approximately 1
        integral = np.trapz(pdf, grid)
        np.testing.assert_allclose(integral, 1.0, atol=0.01)

    def test_default_hyperparams(self, problem):
        hp = problem.default_hyperparams()
        assert 'n_epochs' in hp
        assert 'lam' in hp
        assert 'lip_scale' in hp


class TestQuadraticProblem:
    @pytest.fixture
    def problem(self):
        return QuadraticProblem()

    def test_dimensions(self, problem):
        assert problem.theta_dim == 2
        assert problem.y_dim == 1

    def test_sample_joint_shapes(self, problem):
        n = 100
        theta, y = problem.sample_joint(n)
        assert theta.shape == (n, 2)
        assert y.shape == (n,)

    def test_sample_joint_constraint(self, problem):
        """y should equal theta_1^2 + theta_2."""
        theta, y = problem.sample_joint(100)
        np.testing.assert_allclose(y, theta[:, 0]**2 + theta[:, 1])

    def test_compute_distance(self, problem):
        # Points on parabola y=1: theta_2 = 1 - theta_1^2
        theta_on_parabola = np.array([
            [0.0, 1.0],   # theta_1=0, theta_2=1
            [1.0, 0.0],   # theta_1=1, theta_2=0
            [-1.0, 0.0],  # theta_1=-1, theta_2=0
        ])
        dist = problem.compute_distance(theta_on_parabola, y_obs=1.0)
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

    def test_default_hyperparams_has_quad_features(self, problem):
        hp = problem.default_hyperparams()
        assert hp.get('use_quadratic_features') == True


class TestCircleProblem:
    @pytest.fixture
    def problem(self):
        return CircleProblem()

    def test_dimensions(self, problem):
        assert problem.theta_dim == 2
        assert problem.y_dim == 1

    def test_sample_joint_shapes(self, problem):
        n = 100
        theta, y = problem.sample_joint(n)
        assert theta.shape == (n, 2)
        assert y.shape == (n,)

    def test_sample_joint_constraint(self, problem):
        """y should equal theta_1^2 + theta_2^2."""
        theta, y = problem.sample_joint(100)
        np.testing.assert_allclose(y, theta[:, 0]**2 + theta[:, 1]**2)

    def test_compute_distance(self, problem):
        # Points on circle y=1 (radius 1)
        theta_on_circle = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [np.sqrt(0.5), np.sqrt(0.5)],
        ])
        dist = problem.compute_distance(theta_on_circle, y_obs=1.0)
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

    def test_sample_true_posterior(self, problem):
        samples = problem.sample_true_posterior(y_obs=1.0, n_samples=1000)
        assert samples.shape == (1000, 2)
        # All samples should be on circle of radius 1
        radii = np.sqrt(samples[:, 0]**2 + samples[:, 1]**2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-10)
