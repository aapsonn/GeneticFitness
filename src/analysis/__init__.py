from functools import partial
from typing import Callable

from .correlation_matrix import correlation_matrix
from .correlation_with_fitness import correlation_with_fitness
from .linear_regression import linear_regression
from .random_forest_permutation_importance import random_forest_permutation_importance


def analysis_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    match name:
        case "linear_regression":
            return partial(linear_regression, **kwargs)
        case "correlation_with_fitness":
            return partial(correlation_with_fitness, **kwargs)
        case "random_forest_permutation_importance":
            return partial(random_forest_permutation_importance, **kwargs)
        case "correlation_matrix":
            return partial(correlation_matrix, **kwargs)
        case _:
            raise ValueError(f"Unknown analysis function {name}.")
