import pandas as pd


def remove_non_functional(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Remove non-functional mutations from the dataset."""
    return df[df["fitness"] >= cutoff]  # type: ignore


def remove_non_significant(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Remove non-significant mutations from the dataset."""
    return df[df["p_value"] < cutoff]  # type: ignore
