"""
ANGGOTA 2 (opsional) - Simulator data sensor real-time. Mengirim data ke API.
Jalankan (API harus sudah hidup):
    python -m anggota2_database.simulator --interval 1 --count 50
"""
import argparse
import os
import random
import time

import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")


def make_reading(abnormal_prob=0.15):
    t = random.choices(["L", "M", "H"], weights=[6, 3, 1])[0]
    air = round(random.gauss(300, 2), 1)
    reading = {
        "machine_id": f"M-{random.randint(1, 10):02d}",
        "product_id": f"{t}{random.randint(10000, 99999)}",
        "type": t,
        "air_temp": air,
        "process_temp": round(air + 10 + random.gauss(0, 1), 1),
        "rpm": round(random.gauss(1540, 150)),
        "torque": round(max(3.8, random.gauss(40, 6)), 1),
        "tool_wear": random.randint(0, 250),
    }
    if random.random() < abnormal_prob:  # suntik kondisi bermasalah
        kind = random.choice(["overstrain", "heat", "power"])
        if kind == "overstrain":
            reading["tool_wear"] = random.randint(210, 250)
            reading["torque"] = round(random.uniform(55, 70), 1)
        elif kind == "heat":
            reading["rpm"] = random.randint(1200, 1350)
            reading["process_temp"] = round(air + random.uniform(7.0, 8.4), 1)
        else:
            reading["rpm"] = random.randint(2200, 2600)
            reading["torque"] = round(random.uniform(40, 60), 1)
    return reading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--count", type=int, default=30)
    args = ap.parse_args()
    for i in range(args.count):
        r = make_reading()
        try:
            resp = requests.post(f"{API_URL}/readings", json=r, timeout=10)
            d = resp.json()
            print(f"[{i + 1}] {r['machine_id']} -> {d.get('status')} (p={d.get('defect_probability')})")
        except Exception as e:
            print("Gagal kirim:", e)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
