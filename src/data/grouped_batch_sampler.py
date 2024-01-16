from collections import defaultdict
from random import choice, choices
from typing import Iterable, Sized

from torch.utils.data import Dataset, Sampler


class GroupedBatchSampler(Sampler):
    def __init__(self, data: Dataset, group_key: str, batch_size: int = 16):
        if not isinstance(data, Iterable) or not isinstance(data, Sized):
            raise ValueError("Dataset must be an Iterable and Sized.")

        self.data = data
        self.batch_size = batch_size
        self.length = len(self.data) // self.batch_size

        self.grouped_indices = defaultdict(list)
        for i, item in enumerate(self.data):
            self.grouped_indices[item[0][group_key]].append(i)

        self.keys = list(self.grouped_indices.keys())

    def __len__(self):
        return self.length

    def __iter__(self):
        batches_yielded = 0
        while batches_yielded < self.length:
            key = choice(self.keys)
            group = self.grouped_indices[key]

            if self.batch_size < len(group):
                batch = group
            else:
                batch = choices(group, k=self.batch_size)

            batches_yielded += 1
            yield batch
