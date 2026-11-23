# SmartQC-Flow
Sistem Deteksi Anomali Mesin & Prediksi Cacat Produk pada Lini Produksi
(Python + FastAPI + MongoDB + Machine Learning + Streamlit)

## Struktur
```
smartqc-flow/
├── anggota1_backend/    # Anggota 1: REST API (FastAPI)
├── anggota2_database/   # Anggota 2: MongoDB, import data, agregasi, simulator
├── anggota3_ml/         # Anggota 3: training & prediksi ML (model sudah terlatih di /models)
├── anggota4_frontend/   # Anggota 4: dashboard Streamlit
├── data/                # dataset CSV + generator
├── requirements.txt
├── docker-compose.yml   # opsional: MongoDB lokal
└── .env.example
```

## Cara menjalankan (SEMUA perintah dari folder root `smartqc-flow/`)
```bash
# 0. Persiapan
python -m venv venv
venv\Scripts\activate            # Windows   |  source venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
copy .env.example .env           # Windows   |  cp .env.example .env
# isi MONGO_URI di .env (lokal / MongoDB Atlas). Lokal via Docker: docker compose up -d

# 1. (Anggota 3) latih model - sudah ada hasilnya, jalankan ulang bila ingin
python -m anggota3_ml.train

# 2. (Anggota 2) import dataset ke MongoDB
python -m anggota2_database.import_data --reset --limit 3000

# 3. (Anggota 1) jalankan API  -> http://localhost:8000/docs
uvicorn anggota1_backend.main:app --reload --port 8000

# 4. (Anggota 4) jalankan dashboard (terminal baru) -> http://localhost:8501
streamlit run anggota4_frontend/dashboard.py

# 5. (opsional) simulator data real-time (terminal baru)
python -m anggota2_database.simulator --interval 1 --count 50
```
Catatan: data hasil import (langkah 2) belum punya prediksi. Isi dengan menu **Upload CSV**
di dashboard (pakai `data/ai4i2020_synthetic.csv`) atau jalankan simulator, supaya
Overview/Status Mesin/Alert terisi.

## Tugas tiap anggota
| Anggota | Folder | File | Tugas |
|---|---|---|---|
| 1 Backend & API (Ketua teknis) | `anggota1_backend/` | `main.py`, `schemas.py`, `utils.py` | Endpoint input/ambil data, prediksi, alert; integrasi ML + DB; repo GitHub |
| 2 Database & Data Eng. | `anggota2_database/` | `db.py`, `data_utils.py`, `import_data.py`, `aggregations.py`, `simulator.py` | Skema MongoDB, cleaning & import dataset, agregasi dashboard, simulator |
| 3 Machine Learning | `anggota3_ml/` | `features.py`, `train.py`, `predict.py` | Preprocessing, training RF + Isolation Forest, evaluasi, fungsi `predict()` |
| 4 Front End, Dok. & Artikel | `anggota4_frontend/` | `dashboard.py` | Dashboard, testing akhir, laporan/PPT/artikel, video demo |

## Endpoint API
| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | Cek API & MongoDB |
| POST | `/readings` | Input 1 data sensor, prediksi, simpan, alert otomatis |
| GET | `/readings` | Riwayat data sensor |
| POST | `/upload-csv` | Upload CSV, prediksi massal |
| GET | `/predictions` | Riwayat prediksi |
| GET | `/alerts`, PATCH `/alerts/{id}/resolve` | Daftar & penyelesaian alert |
| GET | `/machines` | Status tiap mesin |
| GET | `/stats/*`, `/model/metrics` | Statistik & metrik model |

## Dataset
`data/ai4i2020_synthetic.csv` adalah dataset SINTETIS (10.000 baris) berformat dan beraturan
kegagalan sama dengan **AI4I 2020 Predictive Maintenance** (UCI). Untuk laporan/artikel sebaiknya
pakai data asli: unduh `ai4i2020.csv` dari UCI atau Kaggle, simpan ke `data/ai4i2020.csv`.
Kode otomatis memakai file asli jika ada. Lalu latih ulang model (`python -m anggota3_ml.train`).

## Skema koleksi MongoDB
- `machines`: machine_id, name, line, created_at
- `sensor_logs`: machine_id, product_id, type, air_temp, process_temp, rpm, torque, tool_wear, actual_failure, timestamp, source
- `predictions`: log_id, machine_id, defect_probability, is_defective, anomaly_score, is_anomaly, status, timestamp
- `alerts`: machine_id, prediction_id, level, message, probability, resolved, timestamp
