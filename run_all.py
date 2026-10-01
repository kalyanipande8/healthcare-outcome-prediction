"""
End-to-end runner: simulate EHR data -> engineer features -> train/evaluate
model -> run the local serving 'uptime' simulation. Prints real numbers at
each stage.
"""
from src.generate_data import build_database
from src.feature_engineering import build_feature_table, get_feature_columns
from src.train_model import train_and_evaluate
from src.uptime_simulation import run_uptime_simulation


def main():
    print("\nSTEP 1/4: Simulating EHR data (local pandas/SQLite stand-in for Databricks ingestion)")
    build_database()

    print("\nSTEP 2/4: Engineering features")
    features_df = build_feature_table()
    n_features = len(get_feature_columns(features_df))
    print(f"  -> {n_features} engineered features produced")

    print("\nSTEP 3/4: Training and evaluating model")
    metrics = train_and_evaluate(features_df)

    print("\nSTEP 4/4: Running local model-serving API + uptime simulation")
    uptime_report = run_uptime_simulation()

    print("\n=== SUMMARY ===")
    print(f"Features engineered: {n_features}")
    print(f"Held-out F1 score:   {metrics['f1_score']}")
    print(f"Held-out ROC AUC:    {metrics['roc_auc']}")
    print(f"Local serving success rate (illustrative 'uptime'): {uptime_report['measured_uptime_pct']}%")


if __name__ == "__main__":
    main()
