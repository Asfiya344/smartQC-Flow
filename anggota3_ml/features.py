"""ANGGOTA 3 - Feature engineering (dipakai bersama oleh train.py dan predict.py)."""
import numpy as np
import pandas as pd

TYPE_MAP = {"L": 0, "M": 1, "H": 2}
FEATURES = ["type_code", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
            "temp_diff", "power", "wear_torque"]
ANOMALY_FEATURES = ["air_temp", "process_temp", "rpm", "torque", "tool_wear", "temp_diff", "power"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type_code"] = out["type"].map(TYPE_MAP).fillna(0).astype(int)
    out["temp_diff"] = out["process_temp"] - out["air_temp"]
    out["power"] = out["torque"] * out["rpm"] * 2 * np.pi / 60  # Watt
    out["wear_torque"] = out["tool_wear"] * out["torque"]
    return out
