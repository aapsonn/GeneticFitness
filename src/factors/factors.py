import pandas as pd
from ViennaRNA import fold

from src.utils.sequences import dna_to_aa


def mutated_amino_acids(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the amino acid sequence corresponding to the mutated nucleotides."""
    df["sequence_aa"] = df["sequence_dna"].apply(dna_to_aa)
    return df


def minimum_free_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the minimum free energy of the mutated sequence."""
    df["minimum_free_energy"] = df["mutated_wildtype_dna"].apply(
        lambda seq: fold(seq)[1]
    )
    return df
