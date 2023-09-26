import pandas as pd
from ViennaRNA import fold

from src.utils.sequences import dna_to_aa


def mutated_amino_acids(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the amino acid sequence corresponding to the mutated nucleotides."""
    df["mutated_amino_acids"] = df["sequence_dna"].apply(dna_to_aa)
    return df


def minimum_free_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the minimum free energy of the mutated sequence."""
    df["minimum_free_energy"] = df["mutated_wildtype_dna"].apply(
        lambda seq: fold(seq)[1]
    )
    return df


def at_content(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the minimum at content of the mutated wildtype."""
    df["at_content"] = df["mutated_wildtype_dna"].apply(
        lambda seq: (seq.count("A") + seq.count("T")) / len(seq)
    )
    return df


def positionwise_mutated_nucleotides(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one column per mutated position."""
    mutation_length = df["sequence_dna"].apply(len).max()
    assert (
        mutation_length == df["sequence_dna"].apply(len).max()
    ), "All mutations should have the same length."

    for position in range(mutation_length):  # type: ignore
        df[f"mutated_dna_{position}"] = df["sequence_dna"].apply(lambda x: x[position])

    return df  # type: ignore
