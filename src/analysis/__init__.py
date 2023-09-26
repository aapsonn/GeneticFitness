from functools import partial
from typing import Callable

from .correlation import correlation
from .linear_regression import linear_regression
from .random_forest_permutation_importance import random_forest_permutation_importance


def analysis_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    match name:
        case "linear_regression":
            return partial(linear_regression, **kwargs)
        case "correlation":
            return partial(correlation, **kwargs)
        case "random_forest_permutation_importance":
            return partial(random_forest_permutation_importance, **kwargs)
        case _:
            raise ValueError(f"Unknown analysis function {name}.")
