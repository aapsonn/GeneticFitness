from functools import partial
from typing import Callable

from .linear_regression import linear_regression


def analysis_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    if name == "linear_regression":
        return partial(linear_regression, **kwargs)
    else:
        raise ValueError(f"Unknown analysis function {name}.")
