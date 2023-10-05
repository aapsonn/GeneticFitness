from datetime import datetime
from pathlib import Path
from random import seed
from shutil import copy, copytree, rmtree

import numpy as np
import seaborn as sns
import torch
import yaml
from loguru import logger
from matplotlib import pyplot as plt

from src.analysis import analysis_factory
from src.data.load_data import load_fitness_data
from src.factors import factor_factory
from src.preprocessing import preprocessing_factory

seed(0)
np.random.seed(0)
torch.manual_seed(0)

sns.set_theme()
sns.set_context("paper")

OUTPUT_PATH = Path("data/output")
HISTORY_PATH = Path("data/output_history")
CONFIG_FILE = Path("src/config.yaml")

# load config file

logger.info("Loading config file.")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

run_name = f"{config['run_name']}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

rmtree(OUTPUT_PATH, ignore_errors=True)
OUTPUT_PATH.mkdir(exist_ok=True)

copy(CONFIG_FILE, OUTPUT_PATH)


logger.add(OUTPUT_PATH / "logs.log")
logger.info("Start run {}.", run_name)

# load data

logger.info("Loading data.")

data = load_fitness_data()

# preprocess data

logger.info("Start preprocessing.")

for preprocessing_step in config["preprocessing"]:
    logger.info("Apply {} preprocessing.", preprocessing_step["name"])
    function = preprocessing_factory(**preprocessing_step)
    data = function(data)

data.to_csv(OUTPUT_PATH / "preprocessed.csv")
logger.info("Preprocessed data written to {}.", OUTPUT_PATH / "preprocessed.csv")

# add factors

logger.info("Start adding factors.")

for factor in config["factors"]:
    logger.info("Add {} factor.", factor["name"])
    function = factor_factory(**factor)
    data = function(data)

data.to_csv(OUTPUT_PATH / "with_factors.csv")
logger.info("Data with factors written to {}.", OUTPUT_PATH / "with_factors.csv")

# run analysis

logger.info("Start analysis.")

for analysis_step in config["analysis"]:
    logger.info("Run {} analysis.", analysis_step["name"])

    function = analysis_factory(output_path=OUTPUT_PATH, **analysis_step)
    function(data)

    plt.clf()

# copy to history

copytree(OUTPUT_PATH, HISTORY_PATH / run_name)
