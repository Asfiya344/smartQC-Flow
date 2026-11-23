
import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from anggota2_database.data_utils import normalize_columns
from anggota3_ml.features import build_features, FEATURES


# ==========================================
# 1. MEMBACA DATASET
# ==========================================

print("=" * 60)
print("MODEL COMPARISON - SMARTQC-FLOW")
print("=" * 60)

df = pd.read_csv("data/ai4i2020.csv")
df = normalize_columns(df)
df = build_features(df)

X = df[FEATURES]
y = df["actual_failure"]

print("\nJUMLAH DATA:")
print("Total data:", len(df))
print("Normal:", (y == 0).sum())
print("Failure:", (y == 1).sum())


# ==========================================
# 2. MENDEFINISIKAN MODEL
# ==========================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ==========================================
# 3. STRATIFIED 5-FOLD CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ==========================================
# 4. EVALUASI SETIAP MODEL
# ==========================================

results = []

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("MENGUJI MODEL:", model_name)
    print("-" * 60)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    result = {
        "Model": model_name,
        "Precision Mean": scores["test_precision"].mean(),
        "Precision Std": scores["test_precision"].std(),
        "Recall Mean": scores["test_recall"].mean(),
        "Recall Std": scores["test_recall"].std(),
        "F1 Mean": scores["test_f1"].mean(),
        "F1 Std": scores["test_f1"].std(),
        "ROC-AUC Mean": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std()
    }

    results.append(result)

    print(f"Precision : {result['Precision Mean']:.4f}")
    print(f"Recall    : {result['Recall Mean']:.4f}")
    print(f"F1-Score  : {result['F1 Mean']:.4f}")
    print(f"ROC-AUC   : {result['ROC-AUC Mean']:.4f}")


# ==========================================
# 5. MENYIMPAN HASIL PERBANDINGAN
# ==========================================

results_df = pd.DataFrame(results)

os.makedirs("anggota3_ml/models", exist_ok=True)

output_path = "anggota3_ml/models/model_comparison_cv.csv"

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("HASIL AKHIR PERBANDINGAN MODEL")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

print("\nHASIL DISIMPAN DI:")
print(output_path)

print("\nMODEL COMPARISON SELESAI!")
