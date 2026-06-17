# prepare.py
# -----------
# This file is all about getting the data ready.
# It loads the CSV files, removes columns that are not useful,
# and builds a few extra columns (features) for each row.
#
# main.py just calls prepare_data() and gets back clean data to train on.

import numpy as np
import pandas as pd

# The name of the id column and the column we want to predict.
ID_COL = "ID"
TARGET = "target"


def load_data():
    # Read the two files Kaggle gives us.
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print("train shape:", train.shape)
    print("test shape:", test.shape)

    return train, test


def remove_constant_columns(X, X_test):
    # A constant column has the same number in every row.
    # It can't help the model, so we drop it.
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() == 1:
            constant_cols.append(col)

    X = X.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

    print("removed", len(constant_cols), "constant columns")
    return X, X_test


def remove_duplicate_columns(X, X_test):
    # Some columns are exact copies of another column.
    # Keeping both is pointless, so we keep only the first one.
    # X.T flips the table so columns become rows, then duplicated()
    # tells us which ones repeat.
    is_duplicate = X.T.duplicated()
    duplicate_cols = X.columns[is_duplicate].tolist()

    X = X.drop(columns=duplicate_cols)
    X_test = X_test.drop(columns=duplicate_cols)

    print("removed", len(duplicate_cols), "duplicate columns")
    return X, X_test


def add_row_features(df):
    # The real meaning of the columns is hidden, so we describe each row
    # with some simple statistics instead.
    df = df.copy()
    values = df.values

    # Basic stats using every value in the row.
    df["row_sum"] = values.sum(axis=1)
    df["row_mean"] = values.mean(axis=1)
    df["row_std"] = values.std(axis=1)
    df["row_min"] = values.min(axis=1)
    df["row_max"] = values.max(axis=1)

    # This data is mostly zeros, so count how many zeros / non-zeros there are.
    df["row_nonzero_count"] = (values != 0).sum(axis=1)
    df["row_zero_count"] = (values == 0).sum(axis=1)

    # Same stats again, but only over the non-zero values.
    # We turn zeros into NaN so numpy ignores them.
    nonzero = np.where(values == 0, np.nan, values)
    df["row_nonzero_mean"] = np.nanmean(nonzero, axis=1)
    df["row_nonzero_std"] = np.nanstd(nonzero, axis=1)
    df["row_nonzero_min"] = np.nanmin(nonzero, axis=1)
    df["row_nonzero_max"] = np.nanmax(nonzero, axis=1)

    # If a row was all zeros, those last stats are NaN. Replace them with 0.
    df = df.fillna(0)
    return df


def prepare_data():
    # This is the function main.py calls. It runs all the steps above
    # in order and returns the clean data.
    train, test = load_data()

    # We predict log(1 + target) instead of target because Kaggle scores
    # us with RMSLE, which is an error measured in log space.
    y = np.log1p(train[TARGET])

    # Save the test ids for the submission file later.
    test_ids = test[ID_COL]

    # Features are everything except the id and the target.
    X = train.drop(columns=[ID_COL, TARGET])
    X_test = test.drop(columns=[ID_COL])

    # Clean the columns.
    X, X_test = remove_constant_columns(X, X_test)
    X, X_test = remove_duplicate_columns(X, X_test)

    # Add the extra row features.
    X = add_row_features(X)
    X_test = add_row_features(X_test)

    print("final train shape:", X.shape)
    print("final test shape:", X_test.shape)

    return X, y, X_test, test_ids
