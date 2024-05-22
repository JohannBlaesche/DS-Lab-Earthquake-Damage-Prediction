"""Cleaning step in the pipeline."""

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer


def clean(*dataframes):
    """Run the cleaning.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        DataFrames to clean.

    Returns
    -------
    list of pd.DataFrame
        Cleaned DataFrames.
    """
    cleaned = []
    for df in dataframes:
        X = df.copy()
        X = clean_single(X)
        cleaned.append(X)

    return cleaned


def clean_single(X: pd.DataFrame) -> pd.DataFrame:
    """Clean the given dataframe.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    X = handle_rare_categoricals(X)  # must be before dtype conversion
    X = remove_rare_binary_cols(X)
    X = remove_columns(X, ["count_floors_pre_eq"])
    X = dtype_conversion(X)
    return X


def get_normalizer():
    """Get the normalization pipeline."""
    normalizer = ColumnTransformer(
        transformers=[
            (
                "percentage_normalizer",
                # equivalent to box-cox for positive values,
                # but it can handle zeros (and negative) as well
                PowerTransformer(method="yeo-johnson"),
                ["area_percentage", "height_percentage"],
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return normalizer


def dtype_conversion(X: pd.DataFrame):
    """Convert columns types."""
    cat_cols = X.select_dtypes(include="object").columns
    binary_cols = [col for col in X.columns if col.startswith("has")]
    geo_levels = [col for col in X.columns if col.startswith("geo_level")]  # noqa: F841

    # X[geo_levels] = X[geo_levels].astype("category")
    X[binary_cols] = X[binary_cols].astype(bool)
    X[cat_cols] = X[cat_cols].astype("category")
    X["count_families"] = pd.cut(
        X["count_families"], [0, 1, 2, 10], right=False, labels=["0", "1", "2+"]
    )

    return X


def handle_rare_categoricals(
    X: pd.DataFrame,
    threshold: float = 0.02,
    method: Literal["replace", "remove"] = "replace",
):
    """Handle rare categories in a categorical column."""
    cols = X.select_dtypes(include="object").columns
    for col in cols:
        counts = X[col].value_counts(normalize=True).sort_values(ascending=False)
        filter_out = counts[counts < threshold].index

        if method == "replace":
            X[col] = X[col].replace(filter_out, "other")
        elif method == "remove":
            X = X.loc[~X[col].isin(filter_out)]
        else:
            raise ValueError("method must be either 'replace' or 'remove'")
    return X


def remove_rare_binary_cols(X: pd.DataFrame, threshold: float = 0.01):
    """Remove binary columns with rare values."""
    binary_cols = [col for col in X.columns if col.startswith("has")]
    for col in binary_cols:
        counts = X[col].value_counts(normalize=True)
        if counts.min() < threshold:
            X = X.drop(columns=[col])
    return X


def remove_columns(X: pd.DataFrame, columns: list):
    """Remove columns from the DataFrame."""
    return X.drop(columns=columns)
