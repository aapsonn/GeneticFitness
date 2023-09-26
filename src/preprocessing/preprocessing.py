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


def extend_mutated_sequence(
    df: pd.DataFrame, wildtype_start_position: int, length: int
) -> pd.DataFrame:
    """Remove non-significant mutations from the dataset."""
    df["mutated_wildtype_dna"] = df["sequence_dna"].apply(
        partial(
            get_mutated_subsequence,
            start_position=wildtype_start_position,
            length=length,
        )
    )
    return df  # type: ignore
