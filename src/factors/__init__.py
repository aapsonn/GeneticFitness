from functools import partial
from typing import Callable

from .factors import (
    at_content,
    minimum_free_energy,
    mutated_amino_acids,
    positionwise_mutated_nucleotides,
)


def factor_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    match name:
        case "mutated_amino_acids":
            return partial(mutated_amino_acids, **kwargs)
        case "at_content":
            return partial(at_content, **kwargs)
        case "minimum_free_energy":
            return partial(minimum_free_energy, **kwargs)
        case "positionwise_mutated_nucleotides":
            return partial(positionwise_mutated_nucleotides, **kwargs)
        case _:
            raise ValueError(f"Unknown factor {name}.")
