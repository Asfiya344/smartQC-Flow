"""
ANGGOTA 1 - REST API (FastAPI). Menyatukan MongoDB (Anggota 2) + ML (Anggota 3).

Jalankan dari ROOT project:
    uvicorn anggota1_backend.main:app --reload --port 8000
Dokumentasi interaktif: http://localhost:8000/docs
"""
import io
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from anggota1_backend.schemas import ReadingIn
from anggota1_backend.utils import serialize, to_object_id
from anggota2_database import aggregations as agg
from anggota2_database.data_utils import normalize_columns
from anggota2_database.db import get_db, init_indexes, ping
from anggota3_ml.predict import ALERT_THRESHOLD, predict, predict_batch

app = FastAPI(title="SmartQC-Flow API",
              description="Deteksi anomali mesin & prediksi cacat produk (Python + MongoDB + ML)",
              version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    if ping():
        init_indexes()


def _ensure_machine(machine_id: str):
    get_db().machines.update_one(
        {"machine_id": machine_id},
        {"$setOnInsert": {"machine_id": machine_id, "name": f"Mesin {machine_id}",
                          "line": "Line-1", "created_at": datetime.utcnow()}},
        upsert=True,
    )


def _save_reading(reading: dict, result: dict, source: str) -> dict:
    """Simpan sensor_log + prediction (+ alert bila perlu). Mengembalikan dokumen gabungan."""
    db = get_db()
    now = datetime.utcnow()
    _ensure_machine(reading["machine_id"])

    log = {**reading, "timestamp": now, "source": source}
    log_id = db.sensor_logs.insert_one(log).inserted_id

    pred = {"log_id": log_id, "machine_id": reading["machine_id"], **result, "timestamp": now}
    pred_id = db.predictions.insert_one(pred).inserted_id

    alert_id = None
    if result["defect_probability"] >= ALERT_THRESHOLD or result["status"] == "Bahaya":
        alert = {
            "machine_id": reading["machine_id"], "prediction_id": pred_id,
            "level": "Bahaya",
            "message": f"Risiko cacat/gagal {result['defect_probability']:.0%} pada {reading['machine_id']}",
            "probability": result["defect_probability"], "resolved": False, "timestamp": now,
        }
        alert_id = db.alerts.insert_one(alert).inserted_id
    return {"log_id": str(log_id), "prediction_id": str(pred_id),
            "alert_id": str(alert_id) if alert_id else None}


# ------------------------------------------------------------------ health
@app.get("/health")
def health():
    return {"api": "ok", "mongodb": "ok" if ping() else "tidak terhubung",
            "alert_threshold": ALERT_THRESHOLD}


# ------------------------------------------------------------------ readings
@app.post("/readings")
def create_reading(payload: ReadingIn):
    """Input 1 data sensor -> prediksi ML -> simpan ke MongoDB -> alert otomatis."""
    reading = payload.model_dump()
    result = predict(reading)
    ids = _save_reading(reading, result, source="api")
    return {**result, **ids, "machine_id": reading["machine_id"]}


@app.get("/readings")
def list_readings(machine_id: Optional[str] = None, limit: int = Query(100, ge=1, le=2000)):
    q = {"machine_id": machine_id} if machine_id else {}
    docs = get_db().sensor_logs.find(q).sort("timestamp", -1).limit(limit)
    return [serialize(d) for d in docs]


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), max_rows: int = 500):
    """Upload CSV (format AI4I asli atau ternormalisasi). Tiap baris diprediksi & disimpan."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File harus .csv")
    try:
        df = normalize_columns(pd.read_csv(io.BytesIO(await file.read()))).head(max_rows)
    except Exception as e:
        raise HTTPException(400, f"CSV tidak valid: {e}")
    if df.empty:
        raise HTTPException(400, "CSV kosong setelah dibersihkan")

    if "machine_id" not in df.columns:
        df["machine_id"] = [f"M-{(i % 10) + 1:02d}" for i in range(len(df))]
    rows = df.to_dict("records")
    results = predict_batch(rows)

    summary = {"Normal": 0, "Waspada": 0, "Bahaya": 0}
    alerts = 0
    for row, res in zip(rows, results):
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        row.pop("udi", None)
        ids = _save_reading(row, res, source="csv_upload")
        summary[res["status"]] += 1
        alerts += 1 if ids["alert_id"] else 0
    return {"rows_processed": len(rows), "status_summary": summary, "alerts_created": alerts}


# ------------------------------------------------------------------ predictions
@app.get("/predictions")
def list_predictions(machine_id: Optional[str] = None, status: Optional[str] = None,
                     limit: int = Query(100, ge=1, le=2000)):
    q = {}
    if machine_id:
        q["machine_id"] = machine_id
    if status:
        q["status"] = status
    docs = get_db().predictions.find(q).sort("timestamp", -1).limit(limit)
    return [serialize(d) for d in docs]


# ------------------------------------------------------------------ alerts
@app.get("/alerts")
def list_alerts(resolved: Optional[bool] = None, limit: int = Query(100, ge=1, le=1000)):
    q = {} if resolved is None else {"resolved": resolved}
    docs = get_db().alerts.find(q).sort("timestamp", -1).limit(limit)
    return [serialize(d) for d in docs]


@app.patch("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str):
    res = get_db().alerts.update_one({"_id": to_object_id(alert_id)}, {"$set": {"resolved": True}})
    if res.matched_count == 0:
        raise HTTPException(404, "Alert tidak ditemukan")
    return {"alert_id": alert_id, "resolved": True}


# ------------------------------------------------------------------ machines & stats
@app.get("/machines")
def machines():
    return [serialize(m) for m in agg.machine_status_summary()]


@app.get("/stats/overview")
def stats_overview():
    return {"kpi": agg.kpi_overview(), "status_distribution": agg.status_distribution()}


@app.get("/stats/per-machine")
def stats_per_machine():
    return agg.avg_sensor_per_machine()


@app.get("/stats/defects-per-hour")
def stats_defects_per_hour(hours: int = Query(24, ge=1, le=720)):
    return agg.defects_per_hour(hours)


@app.get("/model/metrics")
def model_metrics():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "anggota3_ml" / "models" / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "Model belum dilatih")
    return json.loads(p.read_text())

@app.get("/stats/top-machines-by-alerts")
def stats_top_machines_by_alerts():
    return agg.top_machines_by_alerts()


@app.get("/stats/avg-defect-by-type")
def stats_avg_defect_by_type():
    return agg.avg_defect_probability_by_type()


@app.get("/stats/daily-anomalies")
def stats_daily_anomalies():
    return agg.daily_anomaly_counts()

