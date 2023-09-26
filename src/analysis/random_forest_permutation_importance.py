from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def random_forest_permutation_importance(
    df: pd.DataFrame,
    output_path: Path,
    variables: list[str],
):
    """Calculates the permutation importance of the given variables when predicting
    the fitness with a random forest regressor."""
    y = df["fitness"]

    # we need to encode the categorical variables
    X_numerical = df[variables].select_dtypes(include="number")
    X_categorical = df[variables].select_dtypes(include="object")
    for categorical_column in X_categorical.columns:
        X_categorical[categorical_column] = LabelEncoder().fit_transform(
            X_categorical[categorical_column]
        )
    X = pd.concat([X_numerical, X_categorical], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    permutation_result = permutation_importance(model, X_val, y_val, scoring="r2")
    importances_dict = {
        "importance_mean": permutation_result["importances_mean"],
        "importance_std": permutation_result["importances_std"],
        "variable": variables,
    }

    importances_df = pd.DataFrame(importances_dict)
    importances_df.to_csv(output_path / "random_forest_permutatuon_importances.csv")

    sns.barplot(
        data=importances_df,
        x="importance_mean",
        y="variable",
        xerr=importances_df["importance_std"],
    )
    plt.savefig(
        output_path / "random_forest_permutatuon_importances.png", bbox_inches="tight"
    )
