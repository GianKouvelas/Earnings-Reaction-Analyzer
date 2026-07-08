"""
train_model.py: Trains a classification model to predict whether a stock's
earnings-day price move will exceed its historical implied move.

Step 1: chronological train/test split + baseline evaluation,
before any real model. We need to know what "beating random guessing" means
before we can claim our model is any good.

Run directly:
    python3 -m model.train_model
"""

import pandas as pd
from data_pipeline.build_features import build_feature_dataframe

TEST_SET_FRACTION = 0.2  # last 20% of data (chronologically) held out for testing


def chronological_split(df: pd.DataFrame):
    """
    Splits the DataFrame into train/test sets based on report_date,
    NOT randomly. The test set is the most recent TEST_SET_FRACTION of events.

    This matters because in finance, a random split would let the model
    "see" patterns from events that happened after the ones it's tested on —
    which could never happen in real, live use. Chronological split keeps
    training strictly in the past relative to testing.
    """
    df_sorted = df.sort_values("report_date").reset_index(drop=True)

    split_index = int(len(df_sorted) * (1 - TEST_SET_FRACTION))
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    return train_df, test_df


def baseline_accuracy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    The simplest possible "model": always predict the majority class
    seen in the training set. Any real model needs to beat this to be
    considered useful at all.
    """
    majority_class = train_df["exceeded_implied_move"].mode()[0]
    predictions = pd.Series(majority_class, index=test_df.index)
    accuracy = (predictions == test_df["exceeded_implied_move"]).mean()
    return accuracy, majority_class


def main():
    df = build_feature_dataframe()
    print(f"Total samples: {len(df)}")

    train_df, test_df = chronological_split(df)
    print(f"Train samples: {len(train_df)} (up to {train_df['report_date'].max()})")
    print(f"Test samples:  {len(test_df)} (from {test_df['report_date'].min()} onward)")

    acc, majority_class = baseline_accuracy(train_df, test_df)
    print(f"\nBaseline (always predict '{majority_class}'): {acc:.2%} accuracy")
    print("Any real model needs to clearly beat this to be worth using.")


if __name__ == "__main__":
    main()