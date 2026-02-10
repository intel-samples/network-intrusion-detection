# !/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


def run_hyperparameter_tuning(
    training_data_file: str,
    evaluation_data_file: str,
    model_file: str,
    output_file: str | None = None,
    subsample: bool = False,
    sklearnex: bool = False,
) -> None:
    if sklearnex:
        from sklearnex import patch_sklearn

        patch_sklearn()
    import logging
    import pickle
    import time
    from pathlib import Path

    import polars as pl
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    # This is necessary to import in order to deserialized the saved model
    from helpers import convert_pl_timestamps_to_hours

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Start of NIDS model evaluation")
    logger.info(f"\tTraining data file: {training_data_file}")
    logger.info(f"\tEvaluation data file: {evaluation_data_file}")

    # Loading the data
    if not subsample:
        data_training = pl.read_csv(training_data_file)
        data_evaluation = pl.read_csv(evaluation_data_file)
    else:
        import numpy as np

        def read_csv_subsample(file_path: str) -> pl.DataFrame:
            num_rows = (
                pl.scan_csv(file_path, null_values="").select(pl.len()).collect().item()
            )
            indices_sample = np.random.default_rng(seed=0).choice(
                num_rows, size=15_000, replace=False
            )
            return (
                pl.scan_csv(file_path, null_values="")
                .select(pl.all().gather(indices_sample))
                .collect()
            )

        data_training = read_csv_subsample(training_data_file)
        data_evaluation = read_csv_subsample(evaluation_data_file)
    X_train = data_training.drop("label")
    y_train = data_training["label"]

    logger.info(f"\tTraining data has: {data_training.shape[0]} rows")
    logger.info(f"\tEvaluation data has: {data_evaluation.shape[0]} rows")

    # Here, the transformer pipeline from 'run_training.py' will be reused.
    # This still requires creating a new pipeline object, but the steps
    # do not need to be defined again.
    with open(model_file, "rb") as inp_file:
        old_pipeline = pickle.load(inp_file)

    # Remark: if the transformers had some logic involving the 'y' variable,
    # such as TargetEncoder, one might instead want to put the transformer
    # under the meta-estimator too, as a pipeline containing the model inside it.
    feature_transformer = old_pipeline.named_steps["feature_transformer"]
    tuned_pipeline = Pipeline(
        [
            (
                "feature_transformer",
                feature_transformer,
            ),
            (
                "rf_model_tuned",
                GridSearchCV(
                    estimator=RandomForestClassifier(
                        # These parameters below are fixed for all trials
                        n_estimators=1_000,
                        random_state=0,
                        n_jobs=-1,
                    ),
                    # These parameters will be changed across trials
                    param_grid={
                        "max_depth": [4, 6],
                        "max_features": ["sqrt", 3],
                    },
                    refit=True,
                    n_jobs=1,  # Parallelization happens within each RandomForestClassifier
                ),
            ),
        ]
    )

    # Now fitting the whole pipeline, with tuning included
    logger.info("Now fitting the tuning pipeline...")
    # This will temporarily disable the logger, so as not to print
    # too many messages from sklearnex.
    if sklearnex:
        logging.getLogger("sklearnex").setLevel(logging.WARN)
    time_start = time.time()
    tuned_pipeline.fit(X_train, y_train)
    time_end = time.time()
    if sklearnex:
        logging.getLogger("sklearnex").setLevel(logging.INFO)
    logger.info(
        f"Model tunning plus feature transformers took {time_end - time_start} seconds."
    )

    logger.info("Best hyperparameters:")
    logger.info(tuned_pipeline.named_steps["rf_model_tuned"].best_params_)

    # Calculating predictions on new data
    logger.info("Calcualting model predictions on hold-out data...")
    time_start = time.time()
    model_predictions = tuned_pipeline.predict(data_evaluation)
    time_end = time.time()
    logger.info(f"\tTime to predict: {time_end - time_start}")

    # Evaluating the results on the hold-out data sample
    accuracy = accuracy_score(data_evaluation["label"], model_predictions)
    conf_matrix = confusion_matrix(data_evaluation["label"], model_predictions)
    classes = tuned_pipeline[-1].classes_
    conf_matrix_df = pl.DataFrame(
        conf_matrix, schema=[f"Predict: {cls}" for cls in classes]
    ).with_columns(pl.Series(classes).alias("True class"))

    logger.info("Classification accuracy: %.2f%%" % (accuracy * 100))
    logger.info("Confusion matrix:")
    logger.info(conf_matrix_df)

    # Saving the model for later
    if output_file is not None:
        logger.info(f"Saving tuned pipeline to file: {output_file}")
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as out_file:
            pickle.dump(tuned_pipeline, out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data",
        type=str,
        required=False,
        default="data/2021.01.17.csv",
        help="Path to the input CSV data file for day 2021-01-17, from which the model will be trained and tuned.",
    )
    parser.add_argument(
        "--evaluation_data",
        type=str,
        required=False,
        default="data/2021.01.18.csv",
        help="Path to the input CSV data file for day 2021-01-18, on wich the tuned model will be evaluated.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=False,
        default="saved_models/reference_model_pipeline.pkl",
        help="Path to the saved model pipeline produced by 'run_training.py'.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default="saved_models/tuned_model_pipeline.pkl",
        help="Path where to save the resulting model pipeline.",
    )
    parser.add_argument(
        "--subsample",
        type=bool,
        required=False,
        default=False,
        help="Whether to sub-sample the data for low-RAM setups.",
    )
    parser.add_argument(
        "--sklearnex",
        type=bool,
        required=False,
        default=False,
        help="Whether to use the Extension for scikit-learn or not.",
    )
    flags = parser.parse_args()
    run_hyperparameter_tuning(
        flags.training_data,
        flags.evaluation_data,
        flags.model_file,
        flags.output_file,
        flags.subsample,
        flags.sklearnex,
    )
