"""
train_model.py: Trains a classification model to predict whether a stock's
earnings-day price move will exceed its historical implied move.

Run directly:
    python3 -m model.train_model
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_pipeline.build_features import build_feature_dataframe

TEST_SET_FRACTION = 0.2  # last 20% of data (chronologically) held out for testing
FEATURE_COLUMNS = ["surprise_pct", "implied_move", "market_cap"]  # numeric features
CATEGORICAL_COLUMNS = ["sector"]


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


def preprocess_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prepares features for the model:
    - One-hot encodes the categorical `sector` column
    - Scales numeric features (mean 0, std 1) — Logistic Regression is
      sensitive to feature scale, since it optimizes based on distances/gradients
    - IMPORTANT: the scaler is *fit* only on training data, then applied to
      both train and test. Fitting on test data would leak information about
      its distribution into training — a subtle form of lookahead bias.

    Returns (X_train, X_test, y_train, y_test) ready for the model.
    """
    combined = pd.concat([train_df, test_df], keys=["train", "test"])
    combined_encoded = pd.get_dummies(combined, columns=CATEGORICAL_COLUMNS)

    encoded_feature_cols = [
        c for c in combined_encoded.columns
        if c in FEATURE_COLUMNS or c.startswith(tuple(f"{col}_" for col in CATEGORICAL_COLUMNS))
    ]

    train_encoded = combined_encoded.loc["train"]
    test_encoded = combined_encoded.loc["test"]

    X_train_raw = train_encoded[encoded_feature_cols]
    X_test_raw = test_encoded[encoded_feature_cols]
    y_train = train_encoded["exceeded_implied_move"]
    y_test = test_encoded["exceeded_implied_move"]

    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train[FEATURE_COLUMNS] = scaler.fit_transform(X_train_raw[FEATURE_COLUMNS])
    X_test[FEATURE_COLUMNS] = scaler.transform(X_test_raw[FEATURE_COLUMNS])

    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }

    return model, predictions, metrics


def print_results(name: str, predictions, y_test, metrics: dict):
    print(f"\n--- {name} results ---")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1']:.2%}")
    cm = confusion_matrix(y_test, predictions)
    print(f"Confusion matrix:\n{cm}")


def main():
    df = build_feature_dataframe()
    print(f"Total samples: {len(df)}")

    train_df, test_df = chronological_split(df)
    print(f"Train samples: {len(train_df)} (up to {train_df['report_date'].max()})")
    print(f"Test samples:  {len(test_df)} (from {test_df['report_date'].min()} onward)")

    acc, majority_class = baseline_accuracy(train_df, test_df)
    print(f"\nBaseline (always predict '{majority_class}'): {acc:.2%} accuracy")

    X_train, X_test, y_train, y_test = preprocess_features(train_df, test_df)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg, logreg_preds, logreg_metrics = train_and_evaluate(logreg, X_train, X_test, y_train, y_test)
    print_results("Logistic Regression", logreg_preds, y_test, logreg_metrics)
    print(f"Improvement over baseline: {logreg_metrics['accuracy'] - acc:+.2%}")

    # Random Forest — can capture non-linear relationships/interactions between
    # features that Logistic Regression can't. Worth comparing on a dataset
    # this size, though small data also means it can overfit more easily.
    rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    rf, rf_preds, rf_metrics = train_and_evaluate(rf, X_train, X_test, y_train, y_test)
    print_results("Random Forest", rf_preds, y_test, rf_metrics)
    print(f"Improvement over baseline: {rf_metrics['accuracy'] - acc:+.2%}")

    print("\n--- Summary ---")
    print(f"Baseline:            {acc:.2%}")
    print(f"Logistic Regression: {logreg_metrics['accuracy']:.2%} (F1={logreg_metrics['f1']:.2%})")
    print(f"Random Forest:       {rf_metrics['accuracy']:.2%} (F1={rf_metrics['f1']:.2%})")


if __name__ == "__main__":
    main()