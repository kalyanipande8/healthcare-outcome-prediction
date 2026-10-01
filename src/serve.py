"""
Real local Flask model-serving API for the readmission-risk model.

Honest framing: this is a locally-runnable serving layer used to demonstrate
a working model-deployment pattern (load trained model -> serve predictions
over HTTP -> health check -> basic request monitoring). It is NOT a claim of
a managed cloud production deployment. Run it with:

    python src/serve.py

Endpoints:
    GET  /health            -> service + model health check
    POST /predict           -> real prediction from the trained model
    GET  /monitoring/stats  -> running request/uptime stats for this process
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "readmission_model.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.json"

app = Flask(__name__)

_state = {
    "model": None,
    "feature_columns": None,
    "start_time": time.time(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
}


def load_model():
    if _state["model"] is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python src/train_model.py` first."
            )
        _state["model"] = joblib.load(MODEL_PATH)
        with open(FEATURE_LIST_PATH) as f:
            _state["feature_columns"] = json.load(f)
    return _state["model"], _state["feature_columns"]


@app.route("/health", methods=["GET"])
def health():
    try:
        load_model()
        model_ok = True
    except Exception as exc:  # noqa: BLE001
        model_ok = False
        return jsonify({"status": "unhealthy", "model_loaded": model_ok, "error": str(exc)}), 503
    uptime_seconds = round(time.time() - _state["start_time"], 2)
    return jsonify({
        "status": "healthy",
        "model_loaded": model_ok,
        "uptime_seconds": uptime_seconds,
        "framing": "local Flask serving process (illustrative, not a managed cloud deployment)",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    _state["total_requests"] += 1
    try:
        model, feature_columns = load_model()
        payload = request.get_json(force=True, silent=False)
        if payload is None:
            raise ValueError("Request body must be JSON")

        records = payload if isinstance(payload, list) else [payload]
        df = pd.DataFrame(records)

        missing = [c for c in feature_columns if c not in df.columns]
        for c in missing:
            df[c] = 0.0
        df = df[feature_columns]

        proba = model.predict_proba(df)[:, 1]
        preds = (proba >= 0.5).astype(int)

        _state["successful_requests"] += 1
        return jsonify({
            "predictions": preds.tolist(),
            "readmission_probability": [round(float(p), 4) for p in proba],
            "n_records": len(records),
            "model_type": type(model).__name__,
        }), 200
    except Exception as exc:  # noqa: BLE001
        _state["failed_requests"] += 1
        return jsonify({"error": str(exc)}), 400


@app.route("/monitoring/stats", methods=["GET"])
def monitoring_stats():
    total = _state["total_requests"]
    success = _state["successful_requests"]
    uptime_pct = round(100.0 * success / total, 2) if total > 0 else 100.0
    return jsonify({
        "uptime_seconds": round(time.time() - _state["start_time"], 2),
        "total_requests": total,
        "successful_requests": success,
        "failed_requests": _state["failed_requests"],
        "request_success_rate_pct": uptime_pct,
        "note": "Illustrative local-serving metrics from this process only, not a real production SLA.",
    }), 200


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=5050, debug=False)
