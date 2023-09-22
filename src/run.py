import yaml
from loguru import logger

from src.analysis import analysis_factory
from src.data.load_data import load_fitness_data
from src.factors import factor_factory
from src.preprocessing import preprocessing_factory

# load config file

logger.info("Loading config file.")

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# load data

logger.info("Loading data.")

data = load_fitness_data()

# preprocess data

logger.info("Start preprocessing.")

for preprocessing_step in config["preprocessing"]:
    logger.info("Apply {} preprocessing.", preprocessing_step["name"])
    function = preprocessing_factory(**preprocessing_step)
    data = function(data)

# add factors

logger.info("Start adding factors.")

for factor in config["factors"]:
    logger.info("Add {} factor.", factor["name"])
    function = factor_factory(**factor)
    data = function(data)

# run analysis

logger.info("Start analysis.")

for analysis_step in config["analysis"]:
    logger.info("Run {} analysis.", analysis_step["name"])
    function = analysis_factory(**analysis_step)
    function(data)
