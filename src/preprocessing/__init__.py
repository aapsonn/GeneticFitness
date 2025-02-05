from functools import partial
from typing import Callable

from .preprocessing import (
    categorical_encode_str,
    extend_mutated_sequence,
    remove_non_functional,
    remove_non_significant,
    rna_loops,
    rna_loops_minimum_free_energy,
    mutated_amino_acids,
    subset_27de,
    remove_stop_codons,
    subsample,
)


def preprocessing_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    match name:
        case "remove_non_functional":
            return partial(remove_non_functional, **kwargs)
        case "subset_27de":
            return partial(subset_27de, **kwargs)
        case "remove_non_significant":
            return partial(remove_non_significant, **kwargs)
        case "extend_mutated_sequence":
            return partial(extend_mutated_sequence, **kwargs)
        case "subsample":
            return partial(subsample, **kwargs)
        case "rna_loops":
            return partial(rna_loops, **kwargs)
        case "rna_loops_minimum_free_energy":
            return partial(rna_loops_minimum_free_energy, **kwargs)
        case "categorical_encode_str":
            return partial(categorical_encode_str, **kwargs)
        case "mutated_amino_acids":
            return partial(mutated_amino_acids, **kwargs)
        case "remove_stop_codons":
            return partial(remove_stop_codons, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
