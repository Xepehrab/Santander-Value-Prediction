# Santander Value Prediction Challenge

My solution for the [Santander Value Prediction Challenge](https://www.kaggle.com/competitions/santander-value-prediction-challenge)
on Kaggle.

The goal is to predict an anonymized transaction `target` value for each
customer. The data is "wide" (more columns than rows) and mostly zeros, and
the competition is scored with **RMSLE** (Root Mean Squared Logarithmic Error).

## How it works

1. **Load** `train.csv` and `test.csv`.
2. **Use `log(1 + target)`** instead of the raw target, because the metric is
   RMSLE (error measured in log space).
3. **Clean the columns** — drop columns that are always the same value, and
   drop exact-duplicate columns.
4. **Add row features** — since the real column meanings are hidden, build
   simple per-row statistics (sum, mean, std, min, max, zero counts, etc.).
5. **Train LightGBM** with 5-fold cross-validation and early stopping.
6. **Save** `submission.csv` in the format Kaggle expects.

## Files

| File | What it does |
|------|--------------|
| `prepare.py` | Loads and cleans the data, builds the row features. |
| `main.py` | Trains the model and writes the submission file. |
| `requirements.txt` | The Python packages needed to run the project. |
| `train.csv` / `test.csv` | The competition data (download from Kaggle). |
| `sample_submission.csv` | Example of the submission format. |
| `submission.csv` | The file produced by `main.py` (this is what you upload). |

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure `train.csv`, `test.csv`, and `sample_submission.csv` are in the same
folder (download them from the competition's **Data** page on Kaggle).

## Run

```bash
python main.py
```

`main.py` automatically uses `prepare.py` to get the clean data, trains the
model, prints the cross-validation score, and creates `submission.csv`.

## Submission

Upload the generated `submission.csv` to the competition page on Kaggle.
