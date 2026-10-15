"""
predictor.py: Loads the saved model artifact and makes predictions for new inputs.

Run directly for a quick sanity check:
    python3 -m model.predictor
"""

import joblib
import pandas as pd

from model.train_final import ARTIFACT_PATH


def load_artifact(path=ARTIFACT_PATH) -> dict:
    return joblib.load(path)


def predict(artifact: dict, surprise_pct: float, implied_move: float,
            market_cap: float, sector: str) -> dict:
    """
    Returns the probability that the move will exceed the implied move,
    plus the hard True/False prediction.
    market_cap is expected in billions, same as in the training data.
    """
    feature_columns = artifact["feature_columns"]

    # Start with all zeros so any sector the model has never seen is simply
    # "none of the known sectors" instead of crashing
    row = {col: 0.0 for col in feature_columns}
    row["surprise_pct"] = surprise_pct
    row["implied_move"] = implied_move
    row["market_cap"] = market_cap

    sector_col = f"sector_{sector}"
    if sector_col in row:
        row[sector_col] = 1.0

    X = pd.DataFrame([row])[feature_columns]
    X[artifact["numeric_columns"]] = artifact["scaler"].transform(X[artifact["numeric_columns"]])

    model = artifact["model"]
    true_index = list(model.classes_).index(True)
    probability = float(model.predict_proba(X)[0][true_index])

    return {
        "probability_exceeds_implied_move": round(probability, 4),
        "prediction": probability >= 0.5,
    }


if __name__ == "__main__":
    artifact = load_artifact()
    print(f"Loaded model trained at {artifact['trained_at']} on {artifact['n_samples']} samples")

    result = predict(
        artifact,
        surprise_pct=5.0,
        implied_move=6.0,
        market_cap=4500.0,
        sector="Technology",
    )
    print(result)