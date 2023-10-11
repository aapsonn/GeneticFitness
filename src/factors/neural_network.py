from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import yaml
from lightning.pytorch.loggers import CometLogger
from torch.utils.data import DataLoader

from src.data.df_dataset import DfDataset
from src.data.splitting import create_data_splits
from src.modeling import loss_factory, model_factory, optimizer_factory


def _get_comet_api_key():
    with open("comet_api_key.yaml", "r") as f:
        return yaml.safe_load(f)["key"]


def neural_network(
    df: pd.DataFrame,
    splitting_config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataloader_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    loss_config: dict[str, Any],
    model_config: dict[str, Any],
    trainer_config: dict[str, Any],
):
    """Trains a neural network."""
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

    logger = CometLogger(api_key=_get_comet_api_key())
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

    trainer = pl.Trainer(logger=logger, **trainer_config)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model=model, dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)

    val_prediction = trainer.predict(model=model, dataloaders=val_loader)
    val_prediction = np.concatenate([batch.numpy() for batch in val_prediction])  # type: ignore # noqa: E501

    val_df["prediction"] = val_prediction
    return val_df
