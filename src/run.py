from datetime import datetime
from pathlib import Path
from shutil import copy

import yaml
from loguru import logger

from src.analysis import analysis_factory
from src.data.load_data import load_fitness_data
from src.factors import factor_factory
from src.preprocessing import preprocessing_factory

OUTPUT_PATH = Path("data/processed")
CONFIG_FILE = Path("src/config.yaml")

# load config file

logger.info("Loading config file.")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

run_name = f"{config['run_name']}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
run_path = OUTPUT_PATH / run_name
run_path.mkdir()

copy(CONFIG_FILE, run_path)


logger.add(run_path / "logs.log")

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

data.to_csv(run_path / "preprocessed.csv")
logger.info("Preprocessed data written to {}.", run_path / "preprocessed.csv")

# add factors

logger.info("Start adding factors.")

for factor in config["factors"]:
    logger.info("Add {} factor.", factor["name"])
    function = factor_factory(**factor)
    data = function(data)

data.to_csv(run_path / "with_factors.csv")
logger.info("Data with factors written to {}.", run_path / "with_factors.csv")

# run analysis

logger.info("Start analysis.")

for analysis_step in config["analysis"]:
    logger.info("Run {} analysis.", analysis_step["name"])
    function = analysis_factory(**analysis_step)
    function(data)
