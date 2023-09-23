from Bio.Seq import Seq


def rna_to_dna(rna: str):
    """Converts an RNA sequence to a DNA sequence."""
    return rna.replace("U", "T")


def dna_to_rna(dna: str):
    """Converts a DNA sequence to an RNA sequence."""
    return dna.replace("T", "U")


def dna_to_aa(dna: str):
    """Converts a DNA sequence to an amino acid sequence."""
    return "".join(Seq(dna_to_rna(dna)).translate())
