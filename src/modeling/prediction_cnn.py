from typing import Callable, Iterator

import lightning.pytorch as pl
from torch import Tensor, nn, optim


class PredictionCNN(pl.LightningModule):
    def __init__(
        self,
        loss: Callable[[Tensor, Tensor], Tensor],
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    ):
        super().__init__()
        self.loss = loss
        self.optimizer = optimizer

        self.constant = nn.Parameter(Tensor([0.1]))

    def forward(self, batch):
        x, y = batch
        x = x.squeeze()
        x = x * self.constant
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
