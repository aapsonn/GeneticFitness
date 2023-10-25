from functools import partial
from typing import Callable

from .preprocessing import (
    categorical_encode_str,
    extend_mutated_sequence,
    remove_non_functional,
    remove_non_significant,
    rna_loops,
    subsample,
)


def preprocessing_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    match name:
        case "remove_non_functional":
            return partial(remove_non_functional, **kwargs)
        case "remove_non_significant":
            return partial(remove_non_significant, **kwargs)
        case "extend_mutated_sequence":
            return partial(extend_mutated_sequence, **kwargs)
        case "subsample":
            return partial(subsample, **kwargs)
        case "rna_loops":
            return partial(rna_loops, **kwargs)
        case "categorical_encode_str":
            return partial(categorical_encode_str, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
