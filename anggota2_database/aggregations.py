"""ANGGOTA 2 - Query agregasi untuk dashboard."""
from datetime import datetime, timedelta
from anggota2_database.db import get_db


def machine_status_summary():
    """Status terakhir tiap mesin berdasarkan prediksi terbaru + jumlah alert aktif."""
    db = get_db()
    result = []
    for m in db.machines.find({}, {"_id": 0}).sort("machine_id", 1):
        last = db.predictions.find_one({"machine_id": m["machine_id"]}, sort=[("timestamp", -1)])
        active_alerts = db.alerts.count_documents({"machine_id": m["machine_id"], "resolved": False})
        total = db.predictions.count_documents({"machine_id": m["machine_id"]})
        result.append({
            **m,
            "status": last["status"] if last else "Belum ada data",
            "last_probability": round(last["defect_probability"], 3) if last else None,
            "last_update": last["timestamp"] if last else None,
            "active_alerts": active_alerts,
            "total_predictions": total,
        })
    return result


def kpi_overview():
    db = get_db()
    total_logs = db.sensor_logs.count_documents({})
    total_pred = db.predictions.count_documents({})
    defective = db.predictions.count_documents({"is_defective": True})
    anomalies = db.predictions.count_documents({"is_anomaly": True})
    active_alerts = db.alerts.count_documents({"resolved": False})
    return {
        "total_logs": total_logs,
        "total_predictions": total_pred,
        "defective_predicted": defective,
        "defect_rate": round(defective / total_pred, 4) if total_pred else 0,
        "anomalies": anomalies,
        "active_alerts": active_alerts,
    }


def status_distribution():
    db = get_db()
    pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    return {d["_id"]: d["count"] for d in db.predictions.aggregate(pipeline)}


def avg_sensor_per_machine():
    db = get_db()
    pipeline = [{"$group": {
        "_id": "$machine_id",
        "avg_air_temp": {"$avg": "$air_temp"},
        "avg_process_temp": {"$avg": "$process_temp"},
        "avg_rpm": {"$avg": "$rpm"},
        "avg_torque": {"$avg": "$torque"},
        "avg_tool_wear": {"$avg": "$tool_wear"},
        "n": {"$sum": 1},
    }}, {"$sort": {"_id": 1}}]
    return [{"machine_id": d.pop("_id"), **{k: round(v, 2) for k, v in d.items()}}
            for d in db.sensor_logs.aggregate(pipeline)]


def defects_per_hour(hours: int = 24):
    db = get_db()
    since = datetime.utcnow() - timedelta(hours=hours)
    pipeline = [
        {"$match": {"timestamp": {"$gte": since}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d %H:00", "date": "$timestamp"}},
            "total": {"$sum": 1},
            "defective": {"$sum": {"$cond": ["$is_defective", 1, 0]}},
        }},
        {"$sort": {"_id": 1}},
    ]
    return [{"hour": d["_id"], "total": d["total"], "defective": d["defective"]}
            for d in db.predictions.aggregate(pipeline)]

def top_machines_by_alerts():
    """Mengambil 5 mesin dengan jumlah alert terbanyak."""
    db = get_db()

    pipeline = [
        {
            "$group": {
                "_id": "$machine_id",
                "total_alerts": {"$sum": 1},
            }
        },
        {"$sort": {"total_alerts": -1, "_id": 1}},
        {"$limit": 5},
        {
            "$project": {
                "_id": 0,
                "machine_id": "$_id",
                "total_alerts": 1,
            }
        },
    ]

    return list(db.alerts.aggregate(pipeline))


def avg_defect_probability_by_type():
    """Menghitung rata-rata probabilitas defect berdasarkan tipe produk L, M, H."""
    db = get_db()

    pipeline = [
        {
            "$lookup": {
                "from": "sensor_logs",
                "localField": "log_id",
                "foreignField": "_id",
                "as": "sensor",
            }
        },
        {"$unwind": "$sensor"},
        {
            "$group": {
                "_id": "$sensor.type",
                "avg_defect_probability": {
                    "$avg": "$defect_probability"
                },
                "total_predictions": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    return [
        {
            "type": d["_id"],
            "avg_defect_probability": round(
                d["avg_defect_probability"], 4
            ),
            "total_predictions": d["total_predictions"],
        }
        for d in db.predictions.aggregate(pipeline)
    ]


def daily_anomaly_counts():
    """Menghitung jumlah anomali setiap hari berdasarkan timestamp UTC."""
    db = get_db()

    pipeline = [
        {"$match": {"is_anomaly": True}},
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$timestamp",
                        "timezone": "UTC",
                    }
                },
                "total_anomalies": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    return [
        {
            "date": d["_id"],
            "total_anomalies": d["total_anomalies"],
        }
        for d in db.predictions.aggregate(pipeline)
    ]
