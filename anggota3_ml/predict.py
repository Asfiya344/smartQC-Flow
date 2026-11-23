"""
ANGGOTA 3 - Fungsi prediksi yang dipanggil oleh backend.

    from anggota3_ml.predict import predict
    predict({"type": "M", "air_temp": 300, "process_temp": 310, "rpm": 1500,
             "torque": 40, "tool_wear": 100})
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from anggota3_ml.features import ANOMALY_FEATURES, FEATURES, build_features

MODEL_DIR = Path(__file__).parent / "models"
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.70"))
WARNING_THRESHOLD = 0.40


@lru_cache(maxsize=1)
def _load():
    rf_path = MODEL_DIR / "rf_classifier.joblib"
    if not rf_path.exists():
        raise FileNotFoundError("Model belum dilatih. Jalankan: python -m anggota3_ml.train")
    rf = joblib.load(rf_path)
    iso = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    meta = json.loads((MODEL_DIR / "metrics.json").read_text())
    return rf, iso, meta


def _status(prob: float, is_anomaly: bool) -> str:
    if prob >= ALERT_THRESHOLD:
        return "Bahaya"
    if prob >= WARNING_THRESHOLD or is_anomaly:
        return "Waspada"
    return "Normal"


def predict_batch(rows: list[dict]) -> list[dict]:
    rf, iso, meta = _load()
    df = build_features(pd.DataFrame(rows))
    proba = rf.predict_proba(df[FEATURES])[:, 1]
    raw = -iso.score_samples(df[ANOMALY_FEATURES])
    lo, hi = meta["iso_score_min"], meta["iso_score_max"]
    is_anom = iso.predict(df[ANOMALY_FEATURES]) == -1

    out = []
    for p, s, a in zip(proba, raw, is_anom):
        anomaly_score = float(min(max((s - lo) / (hi - lo + 1e-9), 0.0), 1.0))
        out.append({
            "defect_probability": round(float(p), 4),
            "is_defective": bool(p >= 0.5),
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": bool(a),
            "status": _status(float(p), bool(a)),
        })
    return out


def predict(reading: dict) -> dict:
    return predict_batch([reading])[0]


if __name__ == "__main__":
    contoh = {"type": "L", "air_temp": 300.0, "process_temp": 310.0, "rpm": 1500,
              "torque": 40.0, "tool_wear": 100}
    bahaya = {"type": "L", "air_temp": 300.0, "process_temp": 308.0, "rpm": 1300,
              "torque": 65.0, "tool_wear": 230}
    print("Normal :", predict(contoh))
    print("Bahaya :", predict(bahaya))
