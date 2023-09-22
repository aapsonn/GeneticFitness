from functools import cache
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_PATH = Path("data")
ENTIRE_GENE_PATH = DATA_PATH / "raw" / "J01609.1.fasta"
FITNESS_PATH = DATA_PATH / "raw" / "relative_fitness_data.csv"

MUTATION_POSITION = 692
MUTATION_LENGTH = 9

NON_FUNCTIONAL_FITNESS_CUTOFF = -0.508


@cache
def load_wildtype_gene() -> str:
    """Loads the nucleotide sequence of the wildtype gene from the raw data file."""
    with open(ENTIRE_GENE_PATH, "r") as handle:
        return (
            "".join(line for line in handle.readlines() if not line.startswith(">"))
            .strip()
            .replace("\n", "")
        )


@cache
def load_fitness_data() -> pd.DataFrame:
    """Loads the fitness data corresponding to all mutants."""
    column_mapping = {
        "SV": "sequence_dna",
        "m": "fitness",
        "m_p.value": "p_value",
        "m_se": "standard_error",
    }

    df = pd.read_csv(FITNESS_PATH)
    df = df[["SV", "m", "m_p.value", "m_se"]]
    df = df.rename(columns=column_mapping)  # type: ignore
    return df


@cache
def load_functional_fitness_data() -> pd.DataFrame:
    """Loads the fitness data corresponding to all functional mutants."""
    df = load_fitness_data()
    df = df[df["fitness"] >= NON_FUNCTIONAL_FITNESS_CUTOFF]
    return df  # type: ignore


def get_mutated_subsequence(
    mutation: Optional[str] = None, start_position: int = 593, length: int = 117
) -> str:
    """Loads a subsequence of the entire gene, optionally with a mutation at the
    position where experimentally the gene was altered."""
    sequence = load_wildtype_gene()

    if mutation is not None:
        assert (
            len(mutation) == MUTATION_LENGTH
        ), "Mutation must be of length 9 nucleotides."
        sequence = (
            sequence[:MUTATION_POSITION]
            + mutation
            + sequence[MUTATION_POSITION + MUTATION_LENGTH :]
        )

    sequence = sequence[start_position : start_position + length]

    return sequence
