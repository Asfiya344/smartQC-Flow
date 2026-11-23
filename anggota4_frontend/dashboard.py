"""
ANGGOTA 4 - Dashboard Streamlit.
Jalankan dari ROOT project (API harus hidup):
    streamlit run anggota4_frontend/dashboard.py
"""
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SmartQC-Flow", page_icon="🏭", layout="wide")

STATUS_COLOR = {"Normal": "#2f9e44", "Waspada": "#f59f00", "Bahaya": "#e03131", "Belum ada data": "#868e96"}


def api_get(path, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal terhubung ke API ({API_URL}): {e}")
        return None


def api_send(method, path, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Request gagal: {e}")
        return None


st.sidebar.title("🏭 SmartQC-Flow")
st.sidebar.caption("Deteksi anomali mesin & prediksi cacat produk")
page = st.sidebar.radio("Menu", ["Overview", "Status Mesin", "Grafik Sensor", "Input Data",
                                 "Upload CSV", "Alerts", "Performa Model"])
if st.sidebar.button("🔄 Refresh"):
    st.rerun()
health = api_get("/health")
if health:
    st.sidebar.success(f"API: {health['api']} | MongoDB: {health['mongodb']}")

# =============================================================== OVERVIEW
if page == "Overview":
    st.title("Overview Lini Produksi")
    data = api_get("/stats/overview")
    if data:
        k = data["kpi"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Data Sensor", f"{k['total_logs']:,}")
        c2.metric("Total Prediksi", f"{k['total_predictions']:,}")
        c3.metric("Prediksi Cacat", f"{k['defective_predicted']:,}")
        c4.metric("Defect Rate", f"{k['defect_rate']:.1%}")
        c5.metric("Alert Aktif", k["active_alerts"])

        col1, col2 = st.columns(2)
        dist = data["status_distribution"]
        if dist:
            dfd = pd.DataFrame({"status": list(dist.keys()), "jumlah": list(dist.values())})
            fig = px.pie(dfd, names="status", values="jumlah", hole=0.45, color="status",
                         color_discrete_map=STATUS_COLOR, title="Distribusi Status Prediksi")
            col1.plotly_chart(fig, use_container_width=True)
        hourly = api_get("/stats/defects-per-hour", hours=720)
        if hourly:
            dfh = pd.DataFrame(hourly)
            fig = px.bar(dfh, x="hour", y=["total", "defective"], barmode="group",
                         title="Prediksi per Jam (Total vs Cacat)")
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col2.info("Belum ada prediksi. Kirim data lewat menu Input Data / Upload CSV.")

# =============================================================== STATUS MESIN
elif page == "Status Mesin":
    st.title("Status Mesin")
    machines = api_get("/machines")
    if machines:
        cols = st.columns(5)
        for i, m in enumerate(machines):
            color = STATUS_COLOR.get(m["status"], "#868e96")
            prob = f"{m['last_probability']:.0%}" if m["last_probability"] is not None else "-"
            cols[i % 5].markdown(
                f"""<div style="border-left:6px solid {color};padding:10px 12px;margin-bottom:14px;
                background:rgba(128,128,128,0.08);border-radius:6px">
                <b>{m['machine_id']}</b><br><small>{m.get('name', '')}</small><br>
                <span style="color:{color};font-weight:700">{m['status']}</span><br>
                <small>Risiko: {prob} · Alert: {m['active_alerts']}</small></div>""",
                unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(machines).drop(columns=["last_update"], errors="ignore"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada mesin. Jalankan import data (Anggota 2).")

# =============================================================== GRAFIK SENSOR
elif page == "Grafik Sensor":
    st.title("Grafik Sensor")
    machines = api_get("/machines") or []
    ids = [m["machine_id"] for m in machines]
    c1, c2 = st.columns([1, 1])
    mid = c1.selectbox("Mesin", ["Semua"] + ids)
    limit = c2.slider("Jumlah data terakhir", 20, 500, 150)
    logs = api_get("/readings", machine_id=None if mid == "Semua" else mid, limit=limit)
    if logs:
        df = pd.DataFrame(logs).sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col, title in [("process_temp", "Suhu Proses (K)"), ("rpm", "Kecepatan Putar (rpm)"),
                           ("torque", "Torsi (Nm)"), ("tool_wear", "Tool Wear (menit)")]:
            fig = px.line(df, x="timestamp", y=col, title=title,
                          color="machine_id" if mid == "Semua" else None)
            st.plotly_chart(fig, use_container_width=True)
        preds = api_get("/predictions", machine_id=None if mid == "Semua" else mid, limit=limit)
        if preds:
            dp = pd.DataFrame(preds).sort_values("timestamp")
            dp["timestamp"] = pd.to_datetime(dp["timestamp"])
            fig = px.scatter(dp, x="timestamp", y="defect_probability", color="status",
                             color_discrete_map=STATUS_COLOR, title="Probabilitas Cacat")
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Ambang alert 70%")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada data sensor.")

# =============================================================== INPUT DATA
elif page == "Input Data":
    st.title("Input Data Sensor Manual")
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        machine_id = c1.text_input("ID Mesin", "M-01")
        ptype = c2.selectbox("Tipe Produk (Kualitas)", ["L", "M", "H"])
        product_id = c3.text_input("ID Produk (opsional)", "")
        c1, c2 = st.columns(2)
        air = c1.number_input("Suhu udara (K)", 280.0, 330.0, 300.0, 0.1)
        proc = c2.number_input("Suhu proses (K)", 290.0, 350.0, 310.0, 0.1)
        c1, c2, c3 = st.columns(3)
        rpm = c1.number_input("Kecepatan putar (rpm)", 0, 4000, 1500)
        torque = c2.number_input("Torsi (Nm)", 0.0, 100.0, 40.0, 0.1)
        wear = c3.number_input("Tool wear (menit)", 0, 400, 100)
        go = st.form_submit_button("Prediksi & Simpan")
    if go:
        payload = {"machine_id": machine_id, "type": ptype, "air_temp": air, "process_temp": proc,
                   "rpm": rpm, "torque": torque, "tool_wear": wear}
        if product_id:
            payload["product_id"] = product_id
        res = api_send("POST", "/readings", json=payload)
        if res:
            color = STATUS_COLOR[res["status"]]
            st.markdown(f"### Status: <span style='color:{color}'>{res['status']}</span>",
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilitas cacat/gagal", f"{res['defect_probability']:.1%}")
            c2.metric("Skor anomali", f"{res['anomaly_score']:.2f}")
            c3.metric("Anomali?", "Ya" if res["is_anomaly"] else "Tidak")
            if res.get("alert_id"):
                st.error("🚨 Alert dibuat karena risiko di atas ambang batas!")
    with st.expander("Contoh data bermasalah"):
        st.code("Tipe L | Suhu udara 300 | Suhu proses 308 | rpm 1300 | Torsi 65 | Tool wear 230")

# =============================================================== UPLOAD CSV
elif page == "Upload CSV":
    st.title("Upload CSV Data Produksi")
    st.caption("Format kolom AI4I: Type, Air temperature [K], Process temperature [K], "
               "Rotational speed [rpm], Torque [Nm], Tool wear [min]")
    f = st.file_uploader("Pilih file CSV", type=["csv"])
    n = st.number_input("Maksimum baris diproses", 10, 2000, 300)
    if f and st.button("Proses"):
        with st.spinner("Memproses..."):
            res = api_send("POST", f"/upload-csv?max_rows={int(n)}",
                           files={"file": (f.name, f.getvalue(), "text/csv")})
        if res:
            st.success(f"{res['rows_processed']} baris diproses, {res['alerts_created']} alert dibuat")
            st.json(res["status_summary"])

# =============================================================== ALERTS
elif page == "Alerts":
    st.title("Daftar Alert")
    only_active = st.checkbox("Hanya yang belum selesai", True)
    alerts = api_get("/alerts", resolved=False) if only_active else api_get("/alerts")
    if alerts:
        for a in alerts[:50]:
            c1, c2 = st.columns([5, 1])
            icon = "✅" if a["resolved"] else "🚨"
            c1.write(f"{icon} **{a['machine_id']}** — {a['message']}  \n"
                     f"<small>{a['timestamp']}</small>", unsafe_allow_html=True)
            if not a["resolved"] and c2.button("Selesai", key=a["_id"]):
                api_send("PATCH", f"/alerts/{a['_id']}/resolve")
                st.rerun()
    else:
        st.success("Tidak ada alert.")

# =============================================================== PERFORMA MODEL
elif page == "Performa Model":
    st.title("Performa Model Machine Learning")
    m = api_get("/model/metrics")
    if m:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision']:.2%}")
        c3.metric("Recall", f"{m['recall']:.2%}")
        c4.metric("F1", f"{m['f1']:.2%}")
        c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        st.write("Confusion matrix (baris=aktual, kolom=prediksi):")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=["Aktual Normal", "Aktual Gagal"],
                                  columns=["Pred Normal", "Pred Gagal"]))
        st.write("Isolation Forest:", m["isolation_forest"])
        st.caption(f"Dataset: {m['dataset']} | train={m['n_train']} | test={m['n_test']}")
