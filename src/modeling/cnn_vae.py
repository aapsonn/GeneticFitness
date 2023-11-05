from typing import Callable, Iterator

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import IntTensor, Tensor, nn, optim

LOOP_MAPPING = {"(": 0, ".": 1, ")": 2}


class CNN_VAE(pl.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Iterator[nn.Parameter]], optim.Optimizer],
        input_size: int = 117,
        embedding_size: int = 8,
        num_conv_layers: int = 2,
        num_conv_filters: int = 8,
        filter_size: int = 3,
        fully_connected_size: int = 32,
        latent_size: int = 32,
        kld_weight: float = 0.01,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer = optimizer
        self.kld_weight = kld_weight

        self.embedding = nn.Embedding(3, embedding_size)

        # encoder

        encoder_layers = []

        for i in range(num_conv_layers):
            if i == 0:
                layer_input_size = embedding_size
            else:
                layer_input_size = num_conv_filters

            encoder_layers += [
                nn.Conv1d(layer_input_size, num_conv_filters, filter_size),
                nn.ReLU(),
            ]

        encoder_layers += [
            nn.Flatten(),
            nn.Linear(
                num_conv_filters * (input_size - num_conv_layers * (filter_size - 1)),
                fully_connected_size,
            ),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)

        self.mu = nn.Linear(fully_connected_size, latent_size)
        self.var = nn.Linear(fully_connected_size, latent_size)

        # decoder

        decoder_layers = []

        decoder_layers += [
            nn.Linear(
                latent_size,
                num_conv_filters * (input_size - num_conv_layers * (filter_size - 1)),
            ),
            nn.ReLU(),
            nn.Unflatten(
                1, (num_conv_filters, input_size - num_conv_layers * (filter_size - 1))
            ),
        ]

        for _ in range(num_conv_layers - 1):
            decoder_layers += [
                nn.ConvTranspose1d(num_conv_filters, num_conv_filters, filter_size),
                nn.ReLU(),
            ]

        decoder_layers += [
            nn.ConvTranspose1d(num_conv_filters, embedding_size, filter_size),
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, batch):
        x, _ = batch

        loops = x["rna_loops"]

        encoded_loops = [list(map(LOOP_MAPPING.get, loop)) for loop in loops]
        encoded_loops = IntTensor(encoded_loops)

        input_embedding = self.embedding(
            encoded_loops
        )  # (batch_size, seq_len, embedding_dim)
        input_embedding = input_embedding.permute(
            0, 2, 1
        )  # (batch_size, embedding_dim, seq_len)

        encoding = self.encoder(input_embedding)
        mu = self.mu(encoding)
        var = self.var(encoding)

        z = self.reparameterize(mu, var)
        output_embedding = self.decoder(z)

        return input_embedding, output_embedding, mu, var

    def _get_loss(self, batch, loss_prefix=""):
        _, y = batch
        input_embedding, output_embedding, mu, var = self.forward(batch)

        reconstruction_loss = F.mse_loss(output_embedding, input_embedding)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + var - mu**2 - var.exp(), dim=1), dim=0
        )

        loss = reconstruction_loss + self.kld_weight * kld_loss

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
