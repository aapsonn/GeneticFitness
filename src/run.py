import yaml

from src.analysis import analysis_factory
from src.data.load_data import load_fitness_data
from src.factors import factor_factory
from src.preprocessing import preprocessing_factory

# load config file

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# load data

data = load_fitness_data()

# preprocess data

for preprocessing_step in config["preprocessing"]:
    function = preprocessing_factory(**preprocessing_step)
    data = function(data)

# add factors

for factor in config["factors"]:
    function = factor_factory(**factor)
    data = function(data)

# run analysis

for analysis_step in config["analysis"]:
    function = analysis_factory(**analysis_step)
    function(data)
