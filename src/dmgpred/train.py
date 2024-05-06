"""Training step in the pipeline."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from dmgpred.cleaning import get_normalization_pipeline
from dmgpred.featurize import get_encoder


def get_preprocessor(X: pd.DataFrame):
    """Return the preprocessor for the pipeline."""
    normalizer = get_normalization_pipeline()
    encoder = get_encoder(X)
    selector = SelectKBest(k=15, score_func=mutual_info_classif)
    return Pipeline(
        [
            ("normalizer", normalizer),
            ("encoder", encoder),
            ("selector", selector),
        ],
        verbose=False,
    )


def get_pipeline(X: pd.DataFrame, clf=None):
    """Return the training pipeline."""
    if clf is None:
        clf = DummyClassifier(strategy="most_frequent")
    preprocessor = get_preprocessor(X)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", clf),
        ],
        verbose=False,
    )


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train the model on the full dataset.

    This model is used to predict the damage grade of the test data.
    A seperate evaluation is done using cross-validation.
    """
    clf = XGBClassifier()
    pipe = get_pipeline(X_train, clf=clf)
    pipe.fit(X_train, y_train)
    return pipe
