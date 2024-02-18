import pandas as pd
from ostir import run_ostir


def non_wildtype_rbs_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the overall predicted translation initiation rate of the mutated sequence at
    non-wildtype RBS."""

    def get_non_wildtype_rbs_expression(row):
        results = run_ostir(row.mutated_wildtype_dna)
        return sum(r["expression"] for r in results) - 6.702197e02

    df["non_wildtype_rbs_rate"] = df.apply(get_non_wildtype_rbs_expression, axis=1)
    return df


def rbs_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns with the expression rates for each RBS found in the mutated
    sequences."""
    df["rbs_results"] = df.mutated_wildtype_dna.apply(run_ostir)

    all_bfs_starts = {
        sp for r in df.rbs_results for sp in (sps["start_position"] for sps in r)
    }

    for bfs_start in all_bfs_starts:
        df[f"rbs_expression_{bfs_start}"] = df.apply(
            lambda row: next(
                (
                    i["expression"]
                    for i in row["rbs_results"]
                    if i["start_position"] == bfs_start
                ),
                0,
            ),
            axis=1,
        )

    return df
