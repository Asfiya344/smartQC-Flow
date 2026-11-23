"""
ANGGOTA 2 - Import dataset CSV ke MongoDB.
Jalankan dari ROOT project:
    python -m anggota2_database.import_data
    python -m anggota2_database.import_data --csv data/ai4i2020.csv --limit 5000
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from anggota2_database.db import get_db, init_indexes
from anggota2_database.data_utils import normalize_columns

N_MACHINES = 10


def default_csv() -> Path:
    real = Path("data/ai4i2020.csv")
    return real if real.exists() else Path("data/ai4i2020_synthetic.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(default_csv()))
    ap.add_argument("--limit", type=int, default=3000, help="jumlah baris yang diimport")
    ap.add_argument("--reset", action="store_true", help="hapus data lama dulu")
    args = ap.parse_args()

    db = get_db()
    if args.reset:
        for c in ["machines", "sensor_logs", "predictions", "alerts"]:
            db[c].drop()
        print("Koleksi lama dihapus.")
    init_indexes()

    df = normalize_columns(pd.read_csv(args.csv)).head(args.limit)
    print(f"Membaca {len(df)} baris dari {args.csv}")

    # --- machines ---
    now = datetime.utcnow()
    for i in range(1, N_MACHINES + 1):
        mid = f"M-{i:02d}"
        db.machines.update_one(
            {"machine_id": mid},
            {"$setOnInsert": {"machine_id": mid, "name": f"Mesin CNC {i}",
                              "line": f"Line-{(i - 1) // 5 + 1}", "created_at": now}},
            upsert=True,
        )

    # --- sensor_logs (timestamp dibuat mundur supaya grafik terlihat berurutan) ---
    start = now - timedelta(minutes=len(df))
    docs = []
    for idx, r in df.iterrows():
        docs.append({
            "machine_id": f"M-{(idx % N_MACHINES) + 1:02d}",
            "product_id": str(r.get("product_id", f"P{idx}")),
            "type": r["type"],
            "air_temp": float(r["air_temp"]),
            "process_temp": float(r["process_temp"]),
            "rpm": float(r["rpm"]),
            "torque": float(r["torque"]),
            "tool_wear": float(r["tool_wear"]),
            "actual_failure": int(r.get("actual_failure", 0)),
            "timestamp": start + timedelta(minutes=int(idx)),
            "source": "dataset",
        })
    db.sensor_logs.insert_many(docs)
    print(f"Berhasil import {len(docs)} sensor_logs & {N_MACHINES} mesin ke MongoDB.")


if __name__ == "__main__":
    main()
