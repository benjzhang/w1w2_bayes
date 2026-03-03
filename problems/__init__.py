"""Inverse problems for conditional flow learning."""

from .base import BaseProblem
from .linear import LinearProblem
from .quadratic import QuadraticProblem
from .circle import CircleProblem

__all__ = ['BaseProblem', 'LinearProblem', 'QuadraticProblem', 'CircleProblem']

# Registry for easy access by name
PROBLEMS = {
    'linear': LinearProblem,
    'quadratic': QuadraticProblem,
    'circle': CircleProblem,
}

def get_problem(name: str) -> BaseProblem:
    """Get a problem instance by name."""
    if name not in PROBLEMS:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEMS.keys())}")
    return PROBLEMS[name]()
