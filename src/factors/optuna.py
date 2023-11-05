from typing import Callable, Optional

import pandas as pd
from optuna import Trial, create_study


def _build_optuna_config(trial: Trial, config, name: Optional[str] = None):
    if isinstance(config, list):
        return [_build_optuna_config(trial, item) for item in config]

    if isinstance(config, dict) and "sampler" not in config:
        return {k: _build_optuna_config(trial, v, k) for k, v in config.items()}

    if name and isinstance(config, dict) and "sampler" in config:
        match config["sampler"]:
            case "uniform":
                return trial.suggest_float(name, config["min"], config["max"])
            case "log_uniform":
                return trial.suggest_loguniform(name, config["min"], config["max"])
            case "choice":
                return trial.suggest_categorical(name, config["options"])
            case "randint":
                return trial.suggest_int(name, config["min"], config["max"])
            case _:
                raise ValueError(f"Unknown sampler {config['sampler']}")

    return config


def hyperparameter_tuning(
    df: pd.DataFrame,
    factor_name: str,
    num_trials: int,
    factor_factory: Callable[..., Callable],
    **config,
):
    def run_training(trial: Trial):
        run_config: dict = _build_optuna_config(trial, config)  # type: ignore
        run_config["name"] = factor_name
        model = factor_factory(**run_config, return_score=True)
        score = model(df)
        return score

    study = create_study(direction="minimize")
    study.optimize(run_training, num_trials)
