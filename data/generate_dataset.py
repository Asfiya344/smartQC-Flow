"""
Generator dataset sintetis berformat AI4I 2020 Predictive Maintenance.
Kolom & aturan kegagalan MENGIKUTI deskripsi dataset asli (UCI), yaitu:
  TWF : tool wear 200-240 menit (acak)
  HDF : (process temp - air temp) < 8.6 K  DAN  rpm < 1380
  PWF : daya (torque * rpm * 2pi/60) < 3500 W atau > 9000 W
  OSF : tool wear * torque > 11000 (L) / 12000 (M) / 13000 (H)
  RNF : kegagalan acak 0.1%
Dataset asli bisa dipakai: unduh ai4i2020.csv dari UCI/Kaggle lalu simpan sebagai
data/ai4i2020.csv (nama kolom sama, kode lain tidak perlu diubah).
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 10000

types = rng.choice(["L", "M", "H"], size=N, p=[0.6, 0.3, 0.1])
air = np.round(rng.normal(300, 2, N), 1)
proc = np.round(air + 10 + rng.normal(0, 1, N), 1)
rpm = np.round(rng.normal(1500, 170, N) + rng.exponential(60, N)).astype(int)
rpm = np.clip(rpm, 1168, 2886)
torque = np.round(np.clip(rng.normal(40, 6, N) - (rpm - 1540) * 0.03, 3.8, 76.6), 1)
wear = rng.integers(0, 254, N)

power = torque * rpm * 2 * np.pi / 60
limit = np.array([{"L": 11000, "M": 12000, "H": 13000}[t] for t in types])

TWF = ((wear >= 200) & (wear <= 240) & (rng.random(N) < 0.08)).astype(int)
HDF = (((proc - air) < 8.6) & (rpm < 1380)).astype(int)
PWF = ((power < 3500) | (power > 9000)).astype(int)
OSF = ((wear * torque) > limit).astype(int)
RNF = (rng.random(N) < 0.001).astype(int)
fail = ((TWF + HDF + PWF + OSF + RNF) > 0).astype(int)

df = pd.DataFrame({
    "UDI": np.arange(1, N + 1),
    "Product ID": [f"{t}{47000 + i}" for i, t in enumerate(types)],
    "Type": types,
    "Air temperature [K]": air,
    "Process temperature [K]": proc,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": torque,
    "Tool wear [min]": wear,
    "Machine failure": fail,
    "TWF": TWF, "HDF": HDF, "PWF": PWF, "OSF": OSF, "RNF": RNF,
})

out = Path(__file__).parent / "ai4i2020_synthetic.csv"
df.to_csv(out, index=False)
print(f"Tersimpan: {out} | baris={len(df)} | gagal={df['Machine failure'].sum()} "
      f"({df['Machine failure'].mean():.2%})")
