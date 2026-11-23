"""
ANGGOTA 2 - Koneksi & skema MongoDB.

Koleksi:
  machines     : {machine_id, name, line, created_at}
  sensor_logs  : {machine_id, product_id, type, air_temp, process_temp, rpm,
                  torque, tool_wear, actual_failure, timestamp, source}
  predictions  : {log_id, machine_id, defect_probability, is_defective,
                  anomaly_score, is_anomaly, status, timestamp}
  alerts       : {machine_id, prediction_id, level, message, probability,
                  resolved, timestamp}
"""
import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "smartqc_flow")

_client = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    return get_client()[DB_NAME]


def init_indexes():
    """Dipanggil sekali saat startup supaya query cepat."""
    db = get_db()
    db.machines.create_index([("machine_id", ASCENDING)], unique=True)
    db.sensor_logs.create_index([("machine_id", ASCENDING), ("timestamp", DESCENDING)])
    db.predictions.create_index([("machine_id", ASCENDING), ("timestamp", DESCENDING)])
    db.alerts.create_index([("resolved", ASCENDING), ("timestamp", DESCENDING)])


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False
