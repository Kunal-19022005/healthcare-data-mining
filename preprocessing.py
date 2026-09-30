"""
preprocessing.py
----------------
Stage 1 of the Data Mining workflow: loading, cleaning and splitting data.

What this file does:
  1. Loads the raw CSV file
  2. Prints a quick report (shape, missing values, duplicates, invalid zeros)
  3. Treats impossible zeros (e.g. Glucose = 0) as missing values (NaN)
  4. Splits the data into training and testing sets

Note: the missing values are FILLED later, inside the model pipeline in
train.py. Filling them here, before the split, would let information from the
test set leak into training (called "data leakage").

Run this file directly to see the report and save the cleaned CSV:
    python preprocessing.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_raw_data():
    """Read the raw dataset from data/diabetes.csv."""
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.RAW_DATA_PATH}. "
            "Place diabetes.csv inside the data/ folder."
        )
    df = pd.read_csv(config.RAW_DATA_PATH)

    # Make sure all the columns we expect are really in the file
    expected = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing_columns = [col for col in expected if col not in df.columns]
    if missing_columns:
        raise ValueError(f"These columns are missing from the CSV: {missing_columns}")
    return df


def count_invalid_zeros(df):
    """Count how many impossible zeros each column has."""
    return {col: int((df[col] == 0).sum()) for col in config.INVALID_ZERO_COLUMNS}


def clean_data(df):
    """
    Cleaning steps:
      - remove duplicate rows
      - replace impossible zeros with NaN (missing)
    """
    df = df.drop_duplicates().copy()
    for col in config.INVALID_ZERO_COLUMNS:
        df[col] = df[col].replace(0, np.nan)
    return df


def print_report(raw_df, clean_df):
    """Print a simple summary so the student can see what cleaning did."""
    print("=" * 55)
    print("DATASET REPORT")
    print("=" * 55)
    print("Rows, columns (raw)  :", raw_df.shape)
    print("Duplicate rows       :", int(raw_df.duplicated().sum()))
    print("Real NaN in raw file :", int(raw_df.isnull().sum().sum()))

    print("\nClass counts (Outcome: 0 = non-diabetic, 1 = diabetic):")
    print(raw_df[config.TARGET_COLUMN].value_counts().to_string())

    print("\nInvalid zeros found (treated as missing):")
    for col, count in count_invalid_zeros(raw_df).items():
        percent = 100 * count / len(raw_df)
        print(f"  {col:<15} {count:>4}  ({percent:.1f}%)")

    print("\nMissing values AFTER cleaning (will be filled with median later):")
    print(clean_df.isnull().sum().to_string())
    print("=" * 55)


def get_train_test_split():
    """
    Load -> clean -> split into train and test (80/20 from config).
    'stratify' keeps the same diabetic/non-diabetic ratio in both parts.
    """
    df = clean_data(load_raw_data())
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


if __name__ == "__main__":
    config.create_output_folders()
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    print_report(raw_df, clean_df)
    clean_df.to_csv(config.CLEANED_DATA_PATH, index=False)
    print(f"\nCleaned data saved to: {config.CLEANED_DATA_PATH}")
