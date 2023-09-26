from functools import partial
from typing import Callable

from .preprocessing import (
    extend_mutated_sequence,
    remove_non_functional,
    remove_non_significant,
    subsample,
)


def preprocessing_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    if name == "remove_non_functional":
        return partial(remove_non_functional, **kwargs)
    elif name == "remove_non_significant":
        return partial(remove_non_significant, **kwargs)
    elif name == "extend_mutated_sequence":
        return partial(extend_mutated_sequence, **kwargs)
    elif name == "subsample":
        return partial(subsample, **kwargs)
    else:
        raise ValueError(f"Unknown preprocessing function {name}.")
