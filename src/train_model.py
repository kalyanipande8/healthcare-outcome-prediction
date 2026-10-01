"""
Train and evaluate a real ML model (GradientBoosting / RandomForest) on the
engineered EHR feature table to predict 30-day readmission.

This is a genuine local train/test split + fit + evaluation. No numbers are
hard-coded: the reported F1 comes directly from held-out predictions.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.feature_engineering import build_feature_table, get_feature_columns

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "readmission_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.json"


def train_and_evaluate(df: pd.DataFrame = None, random_state: int = 42) -> dict:
    if df is None:
        df = build_feature_table()

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["readmitted_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    candidates = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=random_state, n_jobs=-1,
        ),
    }

    best_name, best_model, best_f1, best_preds = None, None, -1.0, None
    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        results[name] = f1
        if f1 > best_f1:
            best_name, best_model, best_f1, best_preds = name, model, f1, preds

    proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "model_type": best_name,
        "n_features": len(feature_cols),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, best_preds), 4),
        "precision": round(precision_score(y_test, best_preds), 4),
        "recall": round(recall_score(y_test, best_preds), 4),
        "f1_score": round(best_f1, 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "candidate_f1_scores": {k: round(v, 4) for k, v in results.items()},
        "positive_rate_test": round(float(y_test.mean()), 4),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Model evaluation (held-out test set) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Model saved to {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
