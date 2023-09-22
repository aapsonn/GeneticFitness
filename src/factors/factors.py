import pandas as pd

from src.utils.sequences import dna_to_aa


def mutated_amino_acids(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the amino acid sequence corresponding to the mutated nucleotides."""
    df["sequence_aa"] = df["sequence_dna"].apply(dna_to_aa)
    return df
