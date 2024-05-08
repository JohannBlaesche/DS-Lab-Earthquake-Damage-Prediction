"""Main Script of the pipeline."""

import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import dump
from sklearn import set_config

from dmgpred.cleaning import clean
from dmgpred.evaluate import evaluate
from dmgpred.featurize import featurize
from dmgpred.train import train

DATA_PATH = "./data"
OUTPUT_PATH = "./output"
TEST_VALUES_PATH = f"{DATA_PATH}/test_values.csv"
TRAIN_VALUES_PATH = f"{DATA_PATH}/train_values.csv"
TRAIN_LABELS_PATH = f"{DATA_PATH}/train_labels.csv"
SUBMISSION_PATH = f"{OUTPUT_PATH}/Mandalorians_prediction.csv"
INDEX_COL = "building_id"
TARGET = "damage_grade"


@click.command()
@click.option(
    "--add-metrics",
    default=None,
    help="Additional scoring metrics to report in evaluation.",
)
def main(add_metrics):
    """Run the Damage Prediction Pipeline.

    This pipeline consists of four steps, namely cleaning, featurization,
    training and evaluation.

    The cleaned and featurized data is used to train a model on the full dataset and
    to evaluate the model using cross-validation with StratifiedKFold splits.
    The model is then used to predict the
    damage grade of the test data and the results are saved to a CSV file.
    """
    # a simple timer, could use TQDM later on for progress bars
    np.random.seed(0)
    start = time.perf_counter()

    # keep pandas output in transform
    set_config(transform_output="pandas")
    X_test = pd.read_csv(TEST_VALUES_PATH, index_col=INDEX_COL)
    X_train = pd.read_csv(TRAIN_VALUES_PATH, index_col=INDEX_COL)

    # need building id as index here,
    # otherwise it is interpreted as multi-output classification
    y_train = pd.read_csv(TRAIN_LABELS_PATH, index_col=INDEX_COL)
    y_train = y_train[TARGET] - 1

    X_train, X_test = clean(X_train, X_test)
    X_train, X_test = featurize(X_train, X_test)
    model = train(X_train, y_train)
    dump(model, f"{OUTPUT_PATH}/trained_model.pkl")
    y_pred = model.predict(X_test) + 1

    if add_metrics is not None:
        add_metrics = {metric: metric for metric in add_metrics.split(",")}

    _ = evaluate(model, X_train, y_train, additional_scoring=add_metrics)

    Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
    submission = pd.DataFrame({INDEX_COL: X_test.index, TARGET: y_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    end = time.perf_counter()
    print(f"Finished in {end - start: .2f} seconds.")


if __name__ == "__main__":
    main()
