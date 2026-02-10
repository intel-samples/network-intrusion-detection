# !/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


def run_training(
    data_file: str,
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
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

    from helpers import convert_pl_timestamps_to_hours

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Start of NIDS model training")
    logger.info(f"\tInput data file: {data_file}")
    logger.info(f"\tUsing optimized Extension: {sklearnex}")

    if not subsample:
        data = pl.read_csv(data_file, null_values="")
    else:
        import numpy as np

        num_rows = (
            pl.scan_csv(data_file, null_values="").select(pl.len()).collect().item()
        )
        indices_sample = np.random.default_rng(seed=0).choice(
            num_rows, size=15_000, replace=False
        )
        data = (
            pl.scan_csv(data_file, null_values="")
            .select(pl.all().gather(indices_sample))
            .collect()
        )
        del num_rows, indices_sample
    logger.info(f"\tTraining data has: {data.shape[0]} rows")

    # Feature engineering pipeline, as described in the README file
    transf = ColumnTransformer(
        [
            (
                "onehot_enc",
                OneHotEncoder(
                    sparse_output=False,
                    max_categories=10,
                    handle_unknown="infrequent_if_exist",
                ),
                ["proto", "dest_ip", "dest_port", "src_ip", "src_port"],
            ),
            (
                "hour_features",
                Pipeline(
                    [
                        (
                            "cast",
                            FunctionTransformer(convert_pl_timestamps_to_hours),
                        ),
                        (
                            "encode",
                            ColumnTransformer(
                                [
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            sparse_output=False,
                                            max_categories=10,
                                            handle_unknown="infrequent_if_exist",
                                        ),
                                        [True],
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
                ["time_start"],
            ),
            (
                "numeric_features",
                "passthrough",
                [
                    "avg_ipt",
                    "bytes_in",
                    "bytes_out",
                    "entropy",
                    "num_pkts_out",
                    "num_pkts_in",
                    "total_entropy",
                    "duration",
                ],
            ),
        ],
        remainder="drop",
    )

    # Machine learning model, embedded into the pipeline
    classifier_model = Pipeline(
        [
            (
                "feature_transformer",
                transf,
            ),
            (
                "rf_model",
                RandomForestClassifier(
                    max_depth=4,
                    n_estimators=1_000,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X = data.drop("label")
    y = data["label"]

    # Now fitting the model
    logger.info("Now fitting the model pipeline...")
    time_start = time.time()
    classifier_model.fit(X, y)
    time_end = time.time()
    logger.info(f"Model and transformers fitting took {time_end - time_start} seconds.")

    # Saving the model for later
    if output_file is not None:
        logger.info(f"Saving model pipeline to file: {output_file}")
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as out_file:
            pickle.dump(classifier_model, out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=False,
        default="data/2021.01.17.csv",
        help="Path to the input CSV data file for day 2021-01-17.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default="saved_models/reference_model_pipeline.pkl",
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
    run_training(flags.data_file, flags.output_file, flags.subsample, flags.sklearnex)
