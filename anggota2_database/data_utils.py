"""ANGGOTA 2 - Pembersihan & normalisasi kolom dataset (dipakai import, upload CSV, dan ML)."""
import pandas as pd

COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "actual_failure",
}
REQUIRED = ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Terima CSV format AI4I asli maupun format yang sudah dinormalisasi."""
    df = df.rename(columns=COLUMN_MAP).copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df["type"] = df["type"].astype(str).str.upper().str.strip()
    df = df[df["type"].isin(["L", "M", "H"])]
    for c in ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).drop_duplicates()
    if "actual_failure" in df.columns:
        df["actual_failure"] = pd.to_numeric(df["actual_failure"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)
