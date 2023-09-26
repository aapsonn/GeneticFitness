from functools import partial
from typing import Callable

from .correlation import correlation
from .linear_regression import linear_regression


def analysis_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    if name == "linear_regression":
        return partial(linear_regression, **kwargs)
    if name == "correlation":
        return partial(correlation, **kwargs)
    else:
        raise ValueError(f"Unknown analysis function {name}.")
