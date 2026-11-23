
# =====================================================
# SMARTQC-FLOW
# ANGGOTA 3 - EXPLORATORY DATA ANALYSIS
# =====================================================

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

from anggota2_database.data_utils import normalize_columns


# =====================================================
# 1. MEMBACA DATASET
# =====================================================

print("=" * 55)
print("EXPLORATORY DATA ANALYSIS - SMARTQC-FLOW")
print("=" * 55)

# Lokasi dataset
DATA_PATH = "data/ai4i2020.csv"

# Lokasi penyimpanan grafik
OUTPUT_DIR = Path("anggota3_ml/models/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Membaca dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset berhasil dibaca!")
print("Jumlah data:", len(df))


# =====================================================
# 2. NORMALISASI DATASET
# =====================================================

df = normalize_columns(df)

print("\nDataset setelah normalisasi:")
print(df.head())

print("\nNama kolom:")
print(df.columns.tolist())


# =====================================================
# 3. MEMAHAMI STRUKTUR DATA
# =====================================================

print("\nINFORMASI DATASET")
print("-" * 55)

df.info()

print("\nUKURAN DATASET")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDATA DUPLIKAT")
print(df.duplicated().sum())


# =====================================================
# 4. STATISTIK DESKRIPTIF
# =====================================================

features = [
    "air_temp",
    "process_temp",
    "rpm",
    "torque",
    "tool_wear"
]

print("\nSTATISTIK DESKRIPTIF")
print("-" * 55)

print(df[features].describe().round(2))


# =====================================================
# 5. DISTRIBUSI SETIAP FITUR
# =====================================================

print("\nMEMBUAT GRAFIK DISTRIBUSI...")

for feature in features:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[feature],
        bins=30,
        edgecolor="black"
    )

    plt.title(f"Distribusi {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frekuensi")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"distribution_{feature}.png",
        dpi=150
    )

    plt.close()


# Distribusi jenis produk
plt.figure(figsize=(7, 5))

df["type"].value_counts().sort_index().plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Distribusi Jenis Produk")
plt.xlabel("Jenis Produk")
plt.ylabel("Jumlah")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distribution_type.png",
    dpi=150
)

plt.close()


# =====================================================
# 6. BOXPLOT BERDASARKAN MACHINE FAILURE
# =====================================================

print("\nMEMBUAT BOXPLOT...")

for feature in features:

    normal = df.loc[
        df["actual_failure"] == 0,
        feature
    ]

    gagal = df.loc[
        df["actual_failure"] == 1,
        feature
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
    [normal, gagal],
    tick_labels=["Normal", "Gagal"]
)

    plt.title(f"{feature} berdasarkan Machine Failure")
    plt.xlabel("Status Mesin")
    plt.ylabel(feature)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"boxplot_{feature}.png",
        dpi=150
    )

    plt.close()


# =====================================================
# 7. HEATMAP KORELASI
# =====================================================

print("\nMEMBUAT HEATMAP KORELASI...")

correlation = df[
    features + ["actual_failure"]
].corr()

fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))

ax.set_xticklabels(
    correlation.columns,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(correlation.columns)

# Menampilkan angka korelasi
for i in range(len(correlation)):
    for j in range(len(correlation)):
        ax.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

ax.set_title("Correlation Heatmap")

fig.colorbar(image, ax=ax)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=150
)

plt.close()

# Simpan angka korelasi
correlation.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)


# =====================================================
# 8. CLASS IMBALANCE
# =====================================================

print("\nMEMBUAT GRAFIK CLASS IMBALANCE...")

class_counts = (
    df["actual_failure"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Normal", "Gagal"],
    class_counts.values,
    edgecolor="black"
)

plt.title("Distribusi Machine Failure")
plt.xlabel("Status Mesin")
plt.ylabel("Jumlah Data")

for bar, value in zip(bars, class_counts.values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_imbalance.png",
    dpi=150
)

plt.close()


# =====================================================
# 9. MENAMPILKAN PERSENTASE KELAS
# =====================================================

print("\nDISTRIBUSI MACHINE FAILURE")

print(class_counts)

print("\nPERSENTASE MACHINE FAILURE")

print(
    (
        df["actual_failure"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# =====================================================
# 10. SELESAI
# =====================================================

print("\n" + "=" * 55)
print("EDA BERHASIL DISELESAIKAN!")
print("=" * 55)

print("\nLokasi penyimpanan grafik:")
print(OUTPUT_DIR.resolve())

print("\nDaftar hasil grafik:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)
