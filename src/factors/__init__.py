from functools import partial
from typing import Callable

from .factors import (
    at_content,
    minimum_free_energy,
    mutated_amino_acids,
    positionwise_mutated_nucleotides,
)
from .neural_network import neural_network
from .optuna import hyperparameter_tuning
from .rna_fn import rna_fn


def factor_factory(name: str, **kwargs) -> Callable:
    """Factory function for functions that add factors."""
    match name:
        case "mutated_amino_acids":
            return partial(mutated_amino_acids, **kwargs)
        case "at_content":
            return partial(at_content, **kwargs)
        case "minimum_free_energy":
            return partial(minimum_free_energy, **kwargs)
        case "positionwise_mutated_nucleotides":
            return partial(positionwise_mutated_nucleotides, **kwargs)
        case "rna_fn":
            return partial(rna_fn, **kwargs)
        case "neural_network":
            return partial(neural_network, **kwargs)
        case "hyperparameter_tuning":
            return partial(
                hyperparameter_tuning, factor_factory=factor_factory, **kwargs
            )
        case _:
            raise ValueError(f"Unknown factor {name}.")
