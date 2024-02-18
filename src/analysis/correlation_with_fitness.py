from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def correlation_col(variable_name: str):
    """Returns the column name for the correlation."""
    return f"{variable_name}_correlation"


def correlation_p_value_col(variable_name: str):
    """Returns the column name for the correlation p-value."""
    return f"{variable_name}_correlation_p_value"


def correlation_significant_col(variable_name: str):
    """Returns the column name for the column indicating if the correlation
    is significant."""
    return f"{variable_name}_correlation_significant"


def correlation_with_fitness(
    df: pd.DataFrame,
    output_path: Path,
    variables: list[str],
    group_by: Optional[list[str]] = None,
    significant_threshold: float = 0.05,
):
    """Calculates the spearman correlation of the given features with the fitness
    and stores the correlation, average significant correlation and histograms of
    the correlations.

    Args:
        df (pd.DataFrame):
            The dataframe to analyze.
        output_path (th):
            The path where the output gets stored.
        variables (list[str]):
            The correlation for these variables with the fitness will be analyzed.
        group_by (Optional[list[str]], optional):
            If provided, the samples are grouped by these column names and then
            each group is analyzed separatly. Defaults to None.
        significant_threshold (float, optional):
            The threshold used to determine signficance of the calculated
            correlation. Defaults to 0.05.
    """
    # remove columns that are not in the df
    variables = [variable for variable in variables if variable in df.columns]

    # group the dataframe if necessary
    if group_by:
        grouped_df = df.groupby(group_by)
        grouped_dict = {key: group for key, group in grouped_df}
    else:
        grouped_dict = {"all": df}

    # calculate the correlations per group

    grouped_correlations = {}

    for key, group_df in grouped_dict.items():
        correlation_matrix, p_value = stats.spearmanr(
            group_df[["fitness"]],
            group_df[variables],
            axis=0,
        )

        if isinstance(correlation_matrix, np.ndarray):
            grouped_correlations[key] = np.concatenate(
                [correlation_matrix[0][1:], p_value[0][1:]]
            )

        if isinstance(correlation_matrix, float):
            grouped_correlations[key] = np.array([correlation_matrix, p_value])

    correlation_df = pd.DataFrame.from_dict(
        grouped_correlations,
        orient="index",
        columns=[correlation_col(variable) for variable in variables]
        + [correlation_p_value_col(variable) for variable in variables],
    )

    # add flags indicating the significance of the correlaiton

    for variable in variables:
        correlation_df[correlation_significant_col(variable)] = (
            correlation_df[correlation_p_value_col(variable)] < significant_threshold
        )

    correlation_df.to_csv(output_path / "correlation.csv")

    # compute mean significant correlation per group

    average_correlation_dict = {}

    for variable in variables:
        significant_groups = correlation_df[
            correlation_df[correlation_significant_col(variable)]
        ]
        average_correlation_dict[variable] = significant_groups[
            correlation_col(variable)
        ].mean()

    average_correlation_df = pd.DataFrame.from_dict(
        average_correlation_dict,
        orient="index",
    )
    average_correlation_df.to_csv(output_path / "average_significant_correlation.csv")

    # plot correlation histograms

    for variable in variables:
        sns.histplot(
            correlation_df,
            x=correlation_col(variable),
            hue=correlation_significant_col(variable),
        )
        plt.savefig(output_path / f"correlation_{variable}.png", bbox_inches="tight")
        plt.clf()
