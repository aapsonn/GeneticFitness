from typing import Callable, Iterator

import lightning.pytorch as pl
from torch import IntTensor, Tensor, nn, optim

LOOP_MAPPING = {"(": 0, ".": 1, ")": 2}


class PredictionCNN(pl.LightningModule):
    def __init__(
        self,
        loss: Callable[[Tensor, Tensor], Tensor],
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
        input_size: int = 117,
        embedding_size: int = 8,
        num_conv_layers: int = 2,
        num_conv_filters: int = 8,
        fully_connected_size: int = 32,
        filter_size: int = 9,
    ):
        super().__init__()
        self.loss = loss
        self.optimizer = optimizer

        self.embedding = nn.Embedding(3, embedding_size)

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(embedding_size, num_conv_filters, filter_size)
        )
        self.conv_layers += (
            nn.Conv1d(num_conv_filters, num_conv_filters, filter_size)
            for _ in range(num_conv_layers - 1)
        )

        self.fully_connected_layer = nn.Linear(
            num_conv_filters * (input_size - num_conv_layers * (filter_size - 1)),
            fully_connected_size,
        )
        self.fitness_layer = nn.Linear(fully_connected_size, 1)

    def forward(self, batch):
        x, _ = batch

        loops = x["rna_loops"]

        encoded_loops = [list(map(LOOP_MAPPING.get, loop)) for loop in loops]
        encoded_loops = IntTensor(encoded_loops)

        x = self.embedding(encoded_loops)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)

        for layer in self.conv_layers:
            x = layer(x)
            x = nn.functional.relu(x)

        x = x.flatten(start_dim=1)
        x = self.fully_connected_layer(x)
        x = nn.functional.relu(x)

        x = self.fitness_layer(x)
        x = x.squeeze()

        return x

    def _get_loss(self, batch, loss_prefix=""):
        _, y = batch
        x = self.forward(batch)
        loss = self.loss(x, y)

        self.log(f"{loss_prefix}_loss", loss)

        return loss

    def training_step(self, batch, _):
        loss = self._get_loss(batch, "train")
        return loss

    def test_step(self, batch, _):
        loss = self._get_loss(batch, "test")
        return loss

    def validation_step(self, batch, _):
        loss = self._get_loss(batch, "val")
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
