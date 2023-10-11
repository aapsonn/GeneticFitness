from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from src.utils.misc import set_seed


def create_data_splits(
    df: pd.DataFrame,
    train_fraction: float,
    test_fraction: float,
    group_variable: Optional[str] = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe into train, validation and test set.
    Optionally, a group variable can be specified not ensure
    that no group is split across the different sets."""
    set_seed(seed)

    if group_variable is not None:
        train_split = GroupShuffleSplit(n_splits=1, train_size=train_fraction)
        train_idx, no_train_idx = next(train_split.split(df, groups=df[group_variable]))

        df_train = df.iloc[train_idx]
        df_non_train = df.iloc[no_train_idx]

        test_val_split = GroupShuffleSplit(
            n_splits=1, train_size=test_fraction / (1 - train_fraction)
        )
        test_idx, val_idx = next(
            train_split.split(df_non_train, groups=df_non_train[group_variable])
        )

        df_test = df_non_train.iloc[test_idx]
        df_val = df_non_train.iloc[val_idx]

    else:
        train_split = ShuffleSplit(n_splits=1, train_size=train_fraction)
        train_idx, no_train_idx = next(train_split.split(df))

        df_train = df.iloc[train_idx]
        df_non_train = df.iloc[no_train_idx]

        test_val_split = ShuffleSplit(
            n_splits=1, train_size=test_fraction / (1 - train_fraction)
        )
        test_idx, val_idx = next(test_val_split.split(df_non_train))

        df_test = df_non_train.iloc[test_idx]
        df_val = df_non_train.iloc[val_idx]

    return df_train, df_val, df_test
