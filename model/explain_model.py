"""
explain_model.py — Uses SHAP to explain the Random Forest model's predictions.

SHAP (SHapley Additive exPlanations) answers: "for a given prediction, how much
did each feature push the outcome toward True vs False?" This turns the model
from a black box into something we can actually reason about and defend.

Run directly:
    python3 -m model.explain_model
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from data_pipeline.build_features import build_feature_dataframe
from model.train_model import chronological_split, preprocess_features


def main():
    df = build_feature_dataframe()
    train_df, test_df = chronological_split(df)
    X_train, X_test, y_train, y_test = preprocess_features(train_df, test_df)

    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # TreeExplainer is optimized for tree-based models (fast, exact for RF/XGBoost/etc.)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values has one array per class.
    # We look at class index 1 (True = "exceeded implied move").
    if isinstance(shap_values, list):
        shap_values_true_class = shap_values[1]
    else:
        # newer shap versions return a single 3D array (samples, features, classes)
        shap_values_true_class = shap_values[:, :, 1]

    # Global feature importance: average absolute SHAP value per feature,
    # across all test samples — "which features matter most, overall?"
    mean_abs_shap = pd.Series(
        abs(shap_values_true_class).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)

    print("Global feature importance (mean |SHAP value|):\n")
    print(mean_abs_shap.to_string())

    # Save a summary plot — visual version of the same ranking, showing
    # both magnitude and direction (does high surprise_pct push toward
    # True or False, for example)
    plt.figure()
    shap.summary_plot(shap_values_true_class, X_test, show=False)
    plt.tight_layout()
    plt.savefig("model/shap_summary.png", dpi=150)
    print("\nSaved SHAP summary plot to model/shap_summary.png")

    # Explain one individual prediction, as a concrete example for a demo
    sample_idx = 0
    print(f"\n--- Explaining test sample #{sample_idx} ---")
    print(f"Actual outcome: {y_test.iloc[sample_idx]}")
    print(f"Predicted: {model.predict(X_test.iloc[[sample_idx]])[0]}")
    for feature, value in zip(X_test.columns, shap_values_true_class[sample_idx]):
        print(f"  {feature}: SHAP={value:+.4f}")


if __name__ == "__main__":
    main()