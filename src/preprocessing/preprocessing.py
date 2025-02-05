from functools import partial

import pandas as pd
from ViennaRNA import fold

from src.data.load_data import get_mutated_subsequence
from src.utils.sequences import dna_to_aa


def subsample(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Subsample the dataset."""
    return df.sample(frac=fraction)  # type: ignore


def remove_non_functional(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Remove non-functional mutations from the dataset."""
    return df[df["fitness"] >= cutoff]  # type: ignore

def subset_27de(df: pd.DataFrame) -> pd.DataFrame:
    """Subset 27D and 27E (aspartic and glutamic acids), i.e., GA at 4ht and 5th positions of 9 nt genotype string"""
    return df[df["sequence_dna"].str[3:5] == "GA"]  # type: ignore


def remove_non_significant(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Remove non-significant mutations from the dataset."""
    return df[df["p_value"] < cutoff]  # type: ignore


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


def categorical_encode_str(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Converts a column with a string into a sequence of 0 and 1."""
    unique_characters = set(sum(df[column].apply(list), []))  # type: ignore
    print(unique_characters)
    assert (
        len(unique_characters) <= 2
    ), "Only two distinct characters are allowed in 'binary_encode_str'."

    lookup = {character: index for index, character in enumerate(unique_characters)}

    df[f"{column}_binary"] = df[column].apply(
        lambda seq: str([lookup[character] for character in seq])
    )

    return df


def rna_loops(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Uses Vienna to add the predicted RNA loops."""
    df["rna_loops"] = df[column].apply(lambda seq: fold(seq)[0])
    return df

def rna_loops_minimum_free_energy(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Uses Vienna to add the predicted RNA loops and minimum_free_energy."""
    df[["rna_loops", "minimum_free_energy"]] = df[column].apply(lambda seq: pd.Series(fold(seq)[:2]))
    return df


def mutated_amino_acids(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the amino acid sequence corresponding to the mutated nucleotides."""
    df["mutated_amino_acids"] = df["sequence_dna"].apply(dna_to_aa)
    return df

def remove_stop_codons(df: pd.DataFrame) -> pd.DataFrame:
    """Remove amino acid sequences containing stop codons"""
    if "mutated_amino_acids" not in df.columns:
        raise ValueError("Column 'mutated_amino_acids' not found in the DataFrame.")
    return df[~df["mutated_amino_acids"].str.contains(r"\*", na=False)]  # type: ignore