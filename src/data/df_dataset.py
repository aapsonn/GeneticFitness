from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class DfDataset(Dataset):
    """This class wraps a pandas DataFrame into a torch Dataset.
    The dataset consists of (x, y) pairs given by the given columns
    in the wrapped DataFrame."""

    def __init__(
        self, df: pd.DataFrame, variables: list[str], target_variable: str = "fitness"
    ):
        X_df = df[variables]
        Y_df = df[target_variable]

        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        self.Y = torch.tensor(Y_df.values, dtype=torch.float32)

    def __getitem__(self, index) -> Any:
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
