from functools import partial

import pandas as pd

from src.data.load_data import get_mutated_subsequence


def subsample(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Subsample the dataset."""
    return df.sample(frac=fraction)  # type: ignore


def remove_non_functional(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Remove non-functional mutations from the dataset."""
    return df[df["fitness"] >= cutoff]  # type: ignore


def remove_non_significant(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Remove non-significant mutations from the dataset."""
    return df[df["p_value"] < cutoff]  # type: ignore


def add_positionwise_mutations(df: pd.DataFrame) -> pd.DataFrame:
    """Add one column per mutated position."""
    mutation_length = df["sequence_dna"].apply(len).max()
    assert (
        mutation_length == df["sequence_dna"].apply(len).max()
    ), "All mutations should have the same length."

    for position in range(mutation_length):  # type: ignore
        df[f"mutated_dna_{position}"] = df["sequence_dna"].apply(lambda x: x[position])

    return df  # type: ignore


def extend_mutated_sequence(
    df: pd.DataFrame, wildtype_start_position: int, length: int
) -> pd.DataFrame:
    """Add a column containing the wildtype sequence combined with the
    performed mutations."""
    df["mutated_wildtype_dna"] = df["sequence_dna"].apply(
        partial(
            get_mutated_subsequence,
            start_position=wildtype_start_position,
            length=length,
        )
    )
    return df  # type: ignore
