# !/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


def run_evaluation(data_file: str, model_file: str) -> None:
    import logging
    import pickle
    import time

    import polars as pl
    from sklearn.metrics import accuracy_score, confusion_matrix

    from helpers import convert_pl_timestamps_to_hours

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Evaluating model quality")
    logger.info(f"Evaluation data file: {data_file}")
    logger.info(f"Saved model file: {model_file}")

    data = pl.read_csv(data_file)
    with open(model_file, "rb") as inp_file:
        model = pickle.load(inp_file)

    logger.info("Calcualting model predictions...")
    time_start = time.time()
    model_predictions = model.predict(data)
    time_end = time.time()
    logger.info(f"\tTime to predict: {time_end - time_start}")

    accuracy = accuracy_score(data["label"], model_predictions)
    conf_matrix = confusion_matrix(data["label"], model_predictions)
    classes = model.named_steps["rf_model"].classes_
    conf_matrix_df = pl.DataFrame(
        conf_matrix, schema=[f"Predict: {cls}" for cls in classes]
    ).with_columns(pl.Series(classes).alias("True class"))

    logger.info("Classification accuracy: %.2f%%" % float(accuracy * 100))
    logger.info("Confusion matrix:")
    logger.info(conf_matrix_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=False,
        default="data/2021.01.18.csv",
        help="Path to the input CSV data file for day 2021-01-18.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=False,
        default="saved_models/reference_model_pipeline.pkl",
        help="Path to the saved model pipeline produced by 'run_training.py'.",
    )
    flags = parser.parse_args()
    run_evaluation(flags.data_file, flags.model_file)
