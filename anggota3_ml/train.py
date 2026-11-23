"""
ANGGOTA 3 - Training model.
Jalankan dari ROOT project:
    python -m anggota3_ml.train
Output (folder anggota3_ml/models/):
    rf_classifier.joblib, isolation_forest.joblib, metrics.json, confusion_matrix.png,
    feature_importance.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from anggota2_database.data_utils import normalize_columns
from anggota3_ml.features import ANOMALY_FEATURES, FEATURES, build_features

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def default_csv() -> Path:
    real = Path("data/ai4i2020.csv")
    return real if real.exists() else Path("data/ai4i2020_synthetic.csv")


def main(csv_path=None):
    csv_path = Path(csv_path) if csv_path else default_csv()
    df = build_features(normalize_columns(pd.read_csv(csv_path)))
    print(f"Dataset: {csv_path} | {len(df)} baris | gagal={df['actual_failure'].mean():.2%}")

    X, y = df[FEATURES], df["actual_failure"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ---------- 1. Random Forest: prediksi produk cacat / mesin gagal ----------
    rf = RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    proba = rf.predict_proba(X_te)[:, 1]

    metrics = {
        "dataset": str(csv_path),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }
    print(classification_report(y_te, pred, target_names=["Normal", "Gagal/Cacat"]))

    # ---------- 2. Isolation Forest: deteksi anomali (unsupervised, dilatih data normal) ----------
    normal = X_tr[y_tr == 0][ANOMALY_FEATURES]
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    iso.fit(normal)
    iso_pred = (iso.predict(X_te[ANOMALY_FEATURES]) == -1).astype(int)  # 1 = anomali
    metrics["isolation_forest"] = {
        "anomaly_rate_test": round(float(iso_pred.mean()), 4),
        "recall_vs_failure": round(recall_score(y_te, iso_pred), 4),
        "precision_vs_failure": round(precision_score(y_te, iso_pred, zero_division=0), 4),
    }
    # batas skor anomali untuk normalisasi 0..1 di predict.py
    scores = -iso.score_samples(normal)
    metrics["iso_score_min"] = float(scores.min())
    metrics["iso_score_max"] = float(np.percentile(-iso.score_samples(X_te[ANOMALY_FEATURES]), 99.5))

    joblib.dump(rf, MODEL_DIR / "rf_classifier.joblib")
    joblib.dump(iso, MODEL_DIR / "isolation_forest.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- Visualisasi untuk laporan/artikel ----------
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Gagal"]); ax.set_yticks([0, 1], ["Normal", "Gagal"])
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title("Confusion Matrix - Random Forest")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout(); fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax, color="#2b6cb0"); ax.set_title("Feature Importance - Random Forest")
    fig.tight_layout(); fig.savefig(MODEL_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print("Model tersimpan di", MODEL_DIR)


if __name__ == "__main__":
    main()
