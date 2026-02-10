# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


# Helper function - needs to be in global scope for pickle to serialize
def convert_pl_timestamps_to_hours(df: "pl.DataFrame") -> "pl.DataFrame":
    import polars as pl

    return df.with_columns(pl.all().cast(pl.Datetime).dt.hour())
