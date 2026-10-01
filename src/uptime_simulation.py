"""
Real local "uptime" / monitoring simulation for the model-serving API.

Uses Flask's own test client (no separate network process needed) to fire a
batch of real requests -- a mix of valid and deliberately malformed payloads
-- against the real /predict and /health endpoints, and reports the true
success rate observed. This is explicitly framed as an illustrative local
serving exercise, not a real production SLA measurement.
"""
import json
import random
from pathlib import Path

import pandas as pd

from src.serve import app, load_model
from src.feature_engineering import build_feature_table, get_feature_columns

RESULTS_PATH = Path(__file__).resolve().parent.parent / "models" / "uptime_report.json"


def run_uptime_simulation(n_requests: int = 500, malformed_rate: float = 0.03, seed: int = 7) -> dict:
    load_model()
    rng = random.Random(seed)

    df = build_feature_table()
    feature_cols = get_feature_columns(df)
    sample_records = df[feature_cols].sample(n=min(200, len(df)), random_state=seed).to_dict(orient="records")

    client = app.test_client()

    successes, failures = 0, 0
    latencies = []
    for i in range(n_requests):
        health_resp = client.get("/health")
        if health_resp.status_code != 200:
            failures += 1
            continue

        is_malformed = rng.random() < malformed_rate
        if is_malformed:
            payload = {"garbage_field": "not_a_real_feature"}
        else:
            payload = rng.choice(sample_records)

        import time
        t0 = time.perf_counter()
        resp = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        latencies.append(time.perf_counter() - t0)

        expected_ok = not is_malformed  # malformed payload is still handled gracefully (filled w/ zeros), so it can succeed
        if resp.status_code == 200:
            successes += 1
        else:
            failures += 1

    total = successes + failures
    uptime_pct = round(100.0 * successes / total, 3) if total else 0.0
    report = {
        "n_requests": total,
        "successful_requests": successes,
        "failed_requests": failures,
        "measured_uptime_pct": uptime_pct,
        "avg_latency_ms": round(1000 * sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_latency_ms": round(1000 * sorted(latencies)[int(0.95 * len(latencies)) - 1], 3) if latencies else None,
        "disclosure": (
            "Illustrative local-serving metric measured by replaying requests against this "
            "process's Flask test client. It reflects local request-handling reliability, "
            "not a real production/cloud uptime SLA."
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Local serving 'uptime' simulation (illustrative) ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report


if __name__ == "__main__":
    run_uptime_simulation()
