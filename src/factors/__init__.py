from functools import partial
from typing import Callable

from .factors import minimum_free_energy, mutated_amino_acids


def factor_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    match name:
        case "mutated_amino_acids":
            return partial(mutated_amino_acids, **kwargs)
        case "minimum_free_energy":
            return partial(minimum_free_energy, **kwargs)
        case _:
            raise ValueError(f"Unknown factor {name}.")
