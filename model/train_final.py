"""
train_final.py: Trains the final Random Forest on ALL available data and saves
it (together with the scaler and column layout) to disk, so the API can load it
at startup instead of retraining on every request.

Evaluation metrics come from train_model.py (chronological split); this script
is only for producing the deployable artifact.

Run directly:
    python3 -m model.train_final
"""

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_pipeline.build_features import build_feature_dataframe
from model.train_model import FEATURE_COLUMNS, CATEGORICAL_COLUMNS

ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "earnings_model.joblib"


def main():
    df = build_feature_dataframe()
    print(f"Training final model on {len(df)} samples...")

    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)
    sector_columns = [c for c in encoded.columns if c.startswith("sector_")]
    feature_columns = FEATURE_COLUMNS + sector_columns

    X = encoded[feature_columns].copy()
    y = encoded["exceeded_implied_move"]

    scaler = StandardScaler()
    X[FEATURE_COLUMNS] = scaler.fit_transform(X[FEATURE_COLUMNS])

    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, y)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,      # exact column order the model expects
        "numeric_columns": FEATURE_COLUMNS,      # the ones the scaler applies to
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(df),
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved model artifact to {ARTIFACT_PATH}")
    print(f"Feature columns: {feature_columns}")


if __name__ == "__main__":
    main()