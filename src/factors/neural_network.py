from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CometLogger
from torch.utils.data import DataLoader

from src.data.df_dataset import DfDataset
from src.data.splitting import create_data_splits
from src.modeling import loss_factory, model_factory, optimizer_factory
from src.utils.misc import set_seed

torch.set_default_device(torch.device("cpu"))


def _get_comet_api_key():
    with open("comet_api_key.yaml", "r") as f:
        return yaml.safe_load(f)["key"]


def neural_network(
    df: pd.DataFrame,
    project_name,
    splitting_config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataloader_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    loss_config: dict[str, Any],
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
    return_score: bool = False,
):
    """Trains a neural network."""
    set_seed()

    torch.set_default_dtype(torch.float32)

    train_df, test_df, val_df = create_data_splits(df, **splitting_config)

    train_dataset = DfDataset(train_df, **dataset_config)
    test_dataset = DfDataset(test_df, **dataset_config)
    val_dataset = DfDataset(val_df, **dataset_config)

    train_loader = DataLoader(train_dataset, **dataloader_config)

    if "shuffle" in dataloader_config:
        dataloader_config["shuffle"] = False

    test_loader = DataLoader(test_dataset, **dataloader_config)
    val_loader = DataLoader(val_dataset, **dataloader_config)

    loss = loss_factory(**loss_config)
    optimizer = optimizer_factory(**optimizer_config)
    model = model_factory(loss=loss, optimizer=optimizer, **model_config)

    logger = CometLogger(api_key=_get_comet_api_key(), project_name=project_name)
    logger.log_hyperparams(
        {
            **splitting_config,
            **dataset_config,
            **dataloader_config,
            **optimizer_config,
            **loss_config,
            **model_config,
            **trainer_config,
        }
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator="cpu",
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=0.01),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"data/models/{project_name}",
                filename="{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
            ),
        ],
        **trainer_config,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    validation_scores = trainer.validate(model=model, dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)

    if return_score:
        return min(s["val_loss"] for s in validation_scores)

    val_prediction = trainer.predict(model=model, dataloaders=val_loader)
    val_prediction = np.concatenate([batch[2].detach().numpy() for batch in val_prediction])  # type: ignore # noqa: E501

    for i in range(val_prediction.shape[1]):
        val_df[f"prediction_{i}"] = val_prediction[:, i]
    return val_df
