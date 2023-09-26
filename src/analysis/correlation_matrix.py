from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def correlation_matrix(
    df: pd.DataFrame,
    output_path: Path,
    variables: list[str],
):
    """Calculates the calculation matrix between the given variables."""
    X = df[variables]

    correlation = X.corr(method="spearman")  # type: ignore

    correlation.to_csv(output_path / "correlation_matrix.csv")

    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        correlation,
        cmap=cmap,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.5},
    )
    plt.savefig(output_path / "correlation_matrix.png", bbox_inches="tight")
