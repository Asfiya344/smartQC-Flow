
"""
SMARTQC-FLOW - Threshold Analysis

Menganalisis threshold Gradient Boosting menggunakan
out-of-fold predictions pada data training.

Threshold dipilih berdasarkan F1-Score training,
kemudian dievaluasi pada test set yang terpisah.

Jalankan dari root project:
    python -m anggota3_ml.threshold_analysis
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from anggota2_database.data_utils import normalize_columns
from anggota3_ml.features import FEATURES, build_features


# ==========================================
# 1. MEMBACA DATASET
# ==========================================

print("=" * 60)
print("THRESHOLD ANALYSIS - SMARTQC-FLOW")
print("=" * 60)

df = pd.read_csv("data/ai4i2020.csv")
df = normalize_columns(df)
df = build_features(df)

X = df[FEATURES]
y = df["actual_failure"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nJumlah data training:", len(X_tr))
print("Jumlah data testing :", len(X_te))


# ==========================================
# 2. KONFIGURASI GRADIENT BOOSTING
# ==========================================

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# ==========================================
# 3. PREDIKSI OUT-OF-FOLD DATA TRAINING
# ==========================================

print("\nMenghitung prediksi 5-Fold pada data training...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_proba = cross_val_predict(
    model,
    X_tr,
    y_tr,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]


# ==========================================
# 4. MENCARI THRESHOLD TERBAIK
# ==========================================

precision, recall, thresholds = precision_recall_curve(
    y_tr,
    oof_proba
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-9)
)

best_index = np.argmax(f1_values)
best_threshold = float(thresholds[best_index])

print("\nTHRESHOLD TERBAIK BERDASARKAN F1 TRAINING:")
print(f"Threshold : {best_threshold:.4f}")
print(f"Precision : {precision[best_index]:.4f}")
print(f"Recall    : {recall[best_index]:.4f}")
print(f"F1-Score  : {f1_values[best_index]:.4f}")


# ==========================================
# 5. MELATIH MODEL DAN MENGUJI THRESHOLD
# ==========================================

print("\nMelatih model pada seluruh data training...")

model.fit(X_tr, y_tr)

test_proba = model.predict_proba(X_te)[:, 1]

threshold_list = [
    ("Threshold 0.40", 0.40),
    ("Threshold 0.50", 0.50),
    ("Threshold 0.70", 0.70),
    ("Best Threshold", best_threshold)
]

results = []

for name, threshold in threshold_list:

    pred = (test_proba >= threshold).astype(int)

    results.append({
        "Threshold Name": name,
        "Threshold": round(threshold, 4),
        "Precision": round(
            precision_score(y_te, pred, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_te, pred, zero_division=0), 4
        ),
        "F1-Score": round(
            f1_score(y_te, pred, zero_division=0), 4
        )
    })


# ==========================================
# 6. MENAMPILKAN HASIL
# ==========================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("PERBANDINGAN THRESHOLD PADA TEST SET")
print("=" * 60)

print(results_df.to_string(index=False))


# ==========================================
# 7. MENYIMPAN HASIL CSV
# ==========================================

model_dir = Path("anggota3_ml/models")
model_dir.mkdir(exist_ok=True)

csv_path = model_dir / "threshold_analysis.csv"

results_df.to_csv(csv_path, index=False)


# ==========================================
# 8. MENYIMPAN PRECISION-RECALL CURVE
# ==========================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    recall[:-1],
    precision[:-1],
    label="Gradient Boosting"
)

ax.scatter(
    recall[best_index],
    precision[best_index],
    marker="o",
    s=80,
    label=f"Best Threshold: {best_threshold:.4f}"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve - Gradient Boosting")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    model_dir / "precision_recall_curve.png",
    dpi=150
)

plt.close(fig)


# ==========================================
# 9. SELESAI
# ==========================================

print("\nFILE HASIL:")
print(csv_path)
print(model_dir / "precision_recall_curve.png")

print("\nTHRESHOLD ANALYSIS SELESAI!")
