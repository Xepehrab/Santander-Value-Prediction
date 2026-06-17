# main.py
# -------
# This is the main file of the project.
# It gets the clean data from prepare.py, trains a LightGBM model
# with 5-fold cross-validation, and saves the submission file.

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Bring in the data preparation function from the other file.
from prepare import prepare_data

# A fixed number so the results are the same every time we run it.
RANDOM_STATE = 42


def rmsle(y_true, y_pred):
    # Our score. The values are already in log space (we used log1p on the
    # target), so a normal RMSE here is the same as the RMSLE Kaggle reports.
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_model(X, y, X_test):
    # Settings for LightGBM. These are safe starting values.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,    # small steps = slower but usually better
        "num_leaves": 64,         # bigger = more complex trees
        "bagging_fraction": 0.8,  # use 80% of the rows for each tree
        "bagging_freq": 1,
        "feature_fraction": 0.8,  # use 80% of the columns for each tree
        "lambda_l1": 1.0,         # keeps the model from overfitting
        "lambda_l2": 1.0,
        "seed": RANDOM_STATE,
        "num_threads": -1,
        "verbose": -1,            # don't print LightGBM's own logs
    }

    # Split the training data into 5 parts (folds).
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # oof = "out of fold": where we store the prediction for each train row.
    oof = np.zeros(len(X))

    # Where we add up the test predictions from each fold.
    test_preds = np.zeros(len(X_test))

    # Train one model for every fold.
    fold = 0
    for train_idx, valid_idx in folds.split(X):
        fold = fold + 1

        # 4 parts are used for training, 1 part for checking.
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # Put the data into LightGBM's own format.
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        # Train. num_boost_round is the max number of trees, but
        # early_stopping stops early when the score stops improving.
        # This is what makes training much faster.
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        # Predict the validation part and the test set.
        oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
        test_preds = test_preds + model.predict(X_test, num_iteration=model.best_iteration) / 5

        score = rmsle(y_valid, oof[valid_idx])
        print("Fold", fold, "RMSLE:", round(score, 5), "- trees used:", model.best_iteration)

    # The overall score using all the out-of-fold predictions.
    print("\nOverall CV RMSLE:", round(rmsle(y, oof), 5))

    return test_preds


def save_submission(test_ids, test_preds):
    # We trained on log(1 + target), so undo it with exp(x) - 1.
    target = np.expm1(test_preds)

    # The target can never be negative, so set any negative value to 0.
    target = np.clip(target, 0, None)

    submission = pd.DataFrame({"ID": test_ids, "target": target})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


def main():
    # Step 1: get the clean data from prepare.py.
    X, y, X_test, test_ids = prepare_data()

    # Step 2: train the model and predict the test set.
    test_preds = train_model(X, y, X_test)

    # Step 3: save the file we upload to Kaggle.
    save_submission(test_ids, test_preds)


if __name__ == "__main__":
    main()
