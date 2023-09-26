from functools import partial
from typing import Callable

from .preprocessing import (
    add_positionwise_mutations,
    extend_mutated_sequence,
    remove_non_functional,
    remove_non_significant,
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
        case "add_positionwise_mutations":
            return partial(add_positionwise_mutations, **kwargs)
        case "subsample":
            return partial(subsample, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
