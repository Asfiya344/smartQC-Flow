
"""
SMARTQC-FLOW - REPRODUCE ANALYSIS

Analisis tambahan untuk mendukung hasil penelitian:
1. Permutation Feature Importance
2. Perbandingan fitur mentah dan fitur hasil engineering
3. Perbandingan model tanpa dan dengan balanced sample weight
4. Bootstrap confidence interval 95%

Jalankan dari root project:
    python anggota3_ml/reproduce_analysis.py
"""

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


# ==================================================
# 1. MENGATUR ROOT PROJECT DAN IMPORT MODUL
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anggota2_database.data_utils import normalize_columns
from anggota3_ml.features import FEATURES, build_features


# ==================================================
# 2. KONFIGURASI
# ==================================================

DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "anggota3_ml"
    / "models"
    / "reproduce_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 2000

MODEL_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": RANDOM_STATE
}

RAW_FEATURES = [
    "type_code",
    "air_temp",
    "process_temp",
    "rpm",
    "torque",
    "tool_wear"
]


# ==================================================
# 3. FUNGSI BANTU EVALUASI
# ==================================================

def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_proba)
        )
    }


def create_model():
    return GradientBoostingClassifier(**MODEL_PARAMS)


# ==================================================
# 4. MEMBACA DAN MENYIAPKAN DATASET
# ==================================================

print("=" * 65)
print("REPRODUCE ANALYSIS - SMARTQC-FLOW")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}"
    )

print("\nMembaca dataset...")

df = pd.read_csv(DATA_PATH)
df = normalize_columns(df)
df = build_features(df)

required_columns = FEATURES + ["actual_failure"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Kolom berikut tidak ditemukan: {missing_columns}"
    )

df = df.dropna(subset=required_columns).copy()

X = df[FEATURES]
y = df["actual_failure"].astype(int)

print("\nINFORMASI DATASET")
print("-" * 65)
print("Jumlah data       :", len(df))
print("Data normal       :", int((y == 0).sum()))
print("Data failure      :", int((y == 1).sum()))
print("Persentase failure:", f"{y.mean() * 100:.2f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nPEMBAGIAN DATA")
print("-" * 65)
print("Training:", len(X_train))
print("Testing :", len(X_test))


# ==================================================
# 5. BASELINE GRADIENT BOOSTING
# ==================================================

print("\n" + "=" * 65)
print("1. BASELINE GRADIENT BOOSTING")
print("=" * 65)

baseline_model = create_model()
baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)
baseline_proba = baseline_model.predict_proba(X_test)[:, 1]

baseline_metrics = calculate_metrics(
    y_test,
    baseline_pred,
    baseline_proba
)

print("\nMetrik baseline:")
for name, value in baseline_metrics.items():
    print(f"{name:12s}: {value:.4f}")


# ==================================================
# 6. PERMUTATION FEATURE IMPORTANCE
# ==================================================

print("\n" + "=" * 65)
print("2. PERMUTATION FEATURE IMPORTANCE")
print("=" * 65)

print("\nMenghitung permutation importance...")

perm_result = permutation_importance(
    baseline_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std
})

importance_df = importance_df.sort_values(
    by="importance_mean",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "permutation_feature_importance.csv",
    index=False
)

print("\nHASIL PERMUTATION IMPORTANCE")
print(importance_df.to_string(index=False))

plt.figure(figsize=(9, 6))

plt.barh(
    importance_df["feature"],
    importance_df["importance_mean"],
    xerr=importance_df["importance_std"]
)

plt.xlabel("Mean decrease in F1-score")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance - Gradient Boosting")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "permutation_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 7. PERBANDINGAN RAW VS ENGINEERED FEATURES
# ==================================================

print("\n" + "=" * 65)
print("3. RAW FEATURES VS ENGINEERED FEATURES")
print("=" * 65)

feature_sets = {
    "Raw Features": RAW_FEATURES,
    "Engineered Features": FEATURES
}

raw_engineered_results = []

for name, feature_columns in feature_sets.items():

    print(f"\nMelatih model: {name}")

    model = create_model()

    model.fit(
        X_train[feature_columns],
        y_train
    )

    pred = model.predict(
        X_test[feature_columns]
    )

    proba = model.predict_proba(
        X_test[feature_columns]
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        pred,
        proba
    )

    result = {
        "model": name,
        **metrics
    }

    raw_engineered_results.append(result)

    print("Hasil:")
    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:.4f}")


raw_engineered_df = pd.DataFrame(
    raw_engineered_results
)

raw_engineered_df.to_csv(
    OUTPUT_DIR / "raw_vs_engineered.csv",
    index=False
)


# ==================================================
# 8. PERBANDINGAN SAMPLE WEIGHT
# ==================================================

print("\n" + "=" * 65)
print("4. CLASS BALANCED SAMPLE WEIGHT")
print("=" * 65)

print("\nMelatih model tanpa sample weight...")

model_normal = create_model()
model_normal.fit(X_train, y_train)

normal_pred = model_normal.predict(X_test)
normal_proba = model_normal.predict_proba(X_test)[:, 1]

normal_metrics = calculate_metrics(
    y_test,
    normal_pred,
    normal_proba
)

print("\nMelatih model dengan balanced sample weight...")

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model_weighted = create_model()

model_weighted.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

weighted_pred = model_weighted.predict(X_test)
weighted_proba = model_weighted.predict_proba(X_test)[:, 1]

weighted_metrics = calculate_metrics(
    y_test,
    weighted_pred,
    weighted_proba
)

sample_weight_df = pd.DataFrame([
    {
        "model": "Without Sample Weight",
        **normal_metrics
    },
    {
        "model": "Balanced Sample Weight",
        **weighted_metrics
    }
])

print("\nHASIL PERBANDINGAN SAMPLE WEIGHT")
print(sample_weight_df.to_string(index=False))

sample_weight_df.to_csv(
    OUTPUT_DIR / "sample_weight_comparison.csv",
    index=False
)


# ==================================================
# 9. STRATIFIED BOOTSTRAP CONFIDENCE INTERVAL
# ==================================================

print("\n" + "=" * 65)
print("5. STRATIFIED BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 65)

print(
    f"\nMelakukan {N_BOOTSTRAP} bootstrap resampling..."
)

rng = np.random.default_rng(RANDOM_STATE)

# Menggunakan prediksi baseline pada test set.
y_test_array = np.asarray(y_test)
pred_array = np.asarray(baseline_pred)
proba_array = np.asarray(baseline_proba)

normal_indices = np.where(y_test_array == 0)[0]
failure_indices = np.where(y_test_array == 1)[0]

bootstrap_results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": []
}

for i in range(N_BOOTSTRAP):

    sampled_normal = rng.choice(
        normal_indices,
        size=len(normal_indices),
        replace=True
    )

    sampled_failure = rng.choice(
        failure_indices,
        size=len(failure_indices),
        replace=True
    )

    sampled_indices = np.concatenate([
        sampled_normal,
        sampled_failure
    ])

    sampled_y = y_test_array[sampled_indices]
    sampled_pred = pred_array[sampled_indices]
    sampled_proba = proba_array[sampled_indices]

    metrics = calculate_metrics(
        sampled_y,
        sampled_pred,
        sampled_proba
    )

    for metric, value in metrics.items():
        bootstrap_results[metric].append(value)


bootstrap_summary = {}

for metric, values in bootstrap_results.items():

    lower = float(
        np.percentile(values, 2.5)
    )

    upper = float(
        np.percentile(values, 97.5)
    )

    mean_value = float(
        np.mean(values)
    )

    bootstrap_summary[metric] = {
        "mean": mean_value,
        "ci_95_lower": lower,
        "ci_95_upper": upper
    }

    print(
        f"{metric:12s}: "
        f"{mean_value:.4f} "
        f"(95% CI: {lower:.4f} - {upper:.4f})"
    )


# ==================================================
# 10. MENYIMPAN SELURUH HASIL
# ==================================================

print("\n" + "=" * 65)
print("6. MENYIMPAN HASIL ANALISIS")
print("=" * 65)

all_results = {
    "dataset": {
        "file": str(DATA_PATH),
        "total_records": int(len(df)),
        "normal_records": int((y == 0).sum()),
        "failure_records": int((y == 1).sum()),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test))
    },

    "model_parameters": MODEL_PARAMS,

    "baseline_metrics": baseline_metrics,

    "raw_vs_engineered": raw_engineered_results,

    "sample_weight_comparison": [
        {
            "model": "Without Sample Weight",
            **normal_metrics
        },
        {
            "model": "Balanced Sample Weight",
            **weighted_metrics
        }
    ],

    "bootstrap_confidence_interval": bootstrap_summary
}

json_path = OUTPUT_DIR / "reproduce_analysis_results.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        all_results,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\nHASIL TERSIMPAN DI:")
print(OUTPUT_DIR.resolve())

print("\nFILE YANG DIHASILKAN:")
for file_path in sorted(OUTPUT_DIR.iterdir()):
    print("-", file_path.name)

print("\nREPRODUCE ANALYSIS SELESAI!")
