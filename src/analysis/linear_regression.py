import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression


def linear_regression(df: pd.DataFrame, variables: list[str], target: str = "fitness"):
    """Perform linear regression on the given dataframe and the given input
    variables."""
    X_numerical = df[variables].select_dtypes(include="number")
    X_categorical = df[variables].select_dtypes(include="object")
    X_categorical = pd.get_dummies(X_categorical)
    X = pd.concat([X_numerical, X_categorical], axis=1)

    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    coefficients_df = pd.DataFrame(
        {"Variable": model.feature_names_in_, "Coefficient": model.coef_}
    )
    logger.info(coefficients_df)
    logger.info(model.score(X, y))
