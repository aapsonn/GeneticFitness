from typing import Any

import pandas as pd
from torch.utils.data import Dataset


class DfDataset(Dataset):
    """This class wraps a pandas DataFrame into a Dataset.
    The dataset consists of (x, y) pairs given by the given columns
    in the wrapped DataFrame."""

    def __init__(
        self, df: pd.DataFrame, variables: list[str], target_variable: str = "fitness"
    ):
        self.X = df[variables]
        self.Y = df[target_variable].astype("float32")

    def __getitem__(self, index) -> Any:
        return self.X.iloc[index].to_dict(), self.Y.iloc[index]

    def __len__(self):
        return len(self.X)
