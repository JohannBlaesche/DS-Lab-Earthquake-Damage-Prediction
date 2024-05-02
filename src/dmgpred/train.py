"""Training step in the pipeline."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from dmgpred.cleaning import get_normalization_pipeline
from dmgpred.featurize import get_encoder


def get_pipeline(X: pd.DataFrame, clf=None):
    """Return the training pipeline."""
    if clf is None:
        clf = DummyClassifier(strategy="most_frequent")
    normalizer = get_normalization_pipeline()
    encoder = get_encoder(X)
    return Pipeline(
        [
            ("normalizer", normalizer),
            ("encoder", encoder),
            ("clf", clf),
        ],
        verbose=False,
    )


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train the model on the full dataset.

    This model is used to predict the damage grade of the test data.
    A seperate evaluation is done using cross-validation.
    """
    clf = RandomForestClassifier(n_estimators=50, max_depth=3)
    pipe = get_pipeline(X_train, clf=clf)
    pipe.fit(X_train, y_train)
    return pipe
