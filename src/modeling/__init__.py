from functools import partial
from typing import Callable, Iterator

import lightning.pytorch as pl
from torch import Tensor, nn, optim

from .cnn_vae import CNN_VAE
from .prediction_cnn import PredictionCNN


def model_factory(
    name: str,
    loss: Callable[[Tensor, Tensor], Tensor],
    optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
    **kwargs,
) -> pl.LightningModule:
    """Factory function for models."""
    match name:
        case "prediction_cnn":
            return PredictionCNN(loss, optimizer, **kwargs)
        case "cnn_vae":
            return CNN_VAE(optimizer, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")


def loss_factory(name: str, **kwargs) -> Callable[[Tensor, Tensor], Tensor]:
    """Factory function for loss functions."""
    match name:
        case "mse":
            return partial(nn.functional.mse_loss, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")


def optimizer_factory(
    name: str, **kwargs
) -> Callable[[Iterator[nn.Parameter]], optim.Optimizer]:
    """Factory function for optimizers."""
    match name:
        case "adam":
            return partial(optim.Adam, **kwargs)
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
