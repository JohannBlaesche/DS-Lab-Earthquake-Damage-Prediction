"""Evaluation step in the pipeline."""

import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    get_scorer,
    get_scorer_names,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold


def evaluate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    additional_scoring: dict | None = None,
    train_scores=True,
) -> float:
    """Evaluate the model using cross-validation.

    The given model is evaluated with cross-validation, using a
    stratified KFold with the given number of folds. The Matthews
    Correlation Coefficient as well as F1 Micro are reported by default.
    This function can also compute additional scoring metrics passed via
    the additional_scoring parameter. The results are printed to the console
    if verbose is set to True.

    Parameters
    ----------
    model : scikit-learn like model with fit and predict methods.
        The model to evaluate.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    n_folds : int, default=5
        Number of folds for cross-validation, by default 5.
    additional_scoring : dict, optional
        Additional scoring metrics to compute, by default None.
        Must be a dict of str, str or dict of str, callable.
        If a dict of str, str is passed, the value must match a name
        of a sklearn metric. If a dict of str, callable is passed, the
        callable must be a valid scoring function, e.g. one created with
        `make_scorer` from sklearn.metrics.

    Returns
    -------
    results : dict
        The results dict of sklearn.model_selection.cross_validate.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    if additional_scoring is None:
        additional_scoring = {}

    scoring = {
        "F1 Micro": "f1_micro",
        "MCC": make_scorer(matthews_corrcoef, greater_is_better=True),
        **additional_scoring,
    }
    results = cross_validate_custom(
        model, X, y, cv=cv, scoring=scoring, train_scores=train_scores
    )
    for key in scoring:
        scores = results[f"test_{key}"]
        logger.success(f"{key}: {scores.mean(): .4f} (± {scores.std(): .4f})")
        if train_scores:
            scores = results[f"train_{key}"]
            logger.success(f"Train {key}: {scores.mean(): .4f} (± {scores.std(): .4f})")

    return results


def cross_validate_custom(model, X, y, cv, scoring, train_scores=False):
    """
    Perform cross-validation with given scoring metrics.

    Parameters
    ----------
    model : scikit-learn like model with fit and predict methods.
        The model to evaluate.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    cv : cross-validation generator
        A cross-validation iterator with `split` method.
    scoring : dict
        A dictionary where keys are strings representing scoring metric names and values
        are either strings representing predefined scorer names in scikit-learn's
        'metrics' module or scorer functions.

    Returns
    -------
    scores : dict
        A dictionary containing the scores for each fold and each scoring metric.
        Keys are strings in the format 'test_<scoring_metric>', where <scoring_metric>
        is the name of the scoring metric. Values are arrays of shape (n_splits,)
        containing the scores for each fold.
    """
    scores = dict()
    scoring = _check_scoring(scoring)
    for key in scoring:
        scores[f"test_{key}"] = np.zeros(cv.n_splits)
        if train_scores:
            scores[f"train_{key}"] = np.zeros(cv.n_splits)

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        start = time.perf_counter()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)

        for key, scorer in scoring.items():
            score = scorer(model, X_test, y_test)
            scores[f"test_{key}"][i] = score
            if train_scores:
                score_train = scorer(model, X_train, y_train)
                scores[f"train_{key}"][i] = score_train

        end = time.perf_counter()
        logger.info(
            f"MCC in Fold {i+1}: {scores['test_MCC'][i]:.4f} (took {end - start:.2f} seconds)"  # noqa: E501
        )

    return scores


def _check_scoring(scoring: dict) -> dict:
    """Check the scoring dict and return a dict with valid scorers."""
    scoring_ = scoring.copy()
    valid_scorers = get_scorer_names()
    for key, value in scoring.items():
        if callable(value):
            continue
        elif isinstance(value, str):
            if value not in valid_scorers:
                raise ValueError(f"Scorer {value} is not a valid scorer.")
            else:
                scoring_[key] = get_scorer(value)
    return scoring_
