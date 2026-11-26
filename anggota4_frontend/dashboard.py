"""
ANGGOTA 4 - Dashboard Streamlit.
Jalankan dari ROOT project (API harus hidup):
    python -m streamlit run anggota4_frontend/dashboard.py
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SmartQC-Flow",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Warna status (mengikuti tema: Normal biru, Waspada oranye, Bahaya merah)
STATUS_COLOR = {
    "Normal": "#3b82f6",
    "Waspada": "#f59f00",
    "Bahaya": "#ef4444",
    "Belum ada data": "#94a3b8"
}

# Warna untuk badge/pill status
STATUS_PILL = {
    "Normal": ("#dcfce7", "#15803d"),
    "Waspada": ("#ffedd5", "#c2410c"),
    "Bahaya": ("#fee2e2", "#b91c1c"),
    "Belum ada data": ("#e2e8f0", "#475569")
}

# ===============================================================
# TEMA (CSS)
# ===============================================================

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* ---------- Latar utama ---------- */
.stApp {
    background: #f5f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"], [data-testid="stToolbar"] *,
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] *,
[data-testid="collapsedControl"], [data-testid="collapsedControl"] * {
    color: #0f2a4a !important;
}

/* Paksa teks konten gelap agar terbaca di latar terang
   (walau browser/Streamlit memakai mode gelap) */
.stApp .block-container,
.stApp .block-container p,
.stApp .block-container li,
.stApp .block-container label,
.stApp .block-container span,
.stApp .block-container [data-testid="stMarkdownContainer"],
.stApp .block-container [data-testid="stCaptionContainer"] {
    color: #0f2a4a;
}

.stApp .block-container [data-testid="stCaptionContainer"] {
    color: #64748b;
}

.stApp .block-container .pill,
.stApp .block-container .pill span {
    color: inherit;
}

/* Kotak kode / JSON ikut tema terang */
.stApp pre, .stApp code, [data-testid="stJson"], [data-testid="stCode"] {
    background: #f1f5f9 !important;
    color: #0f2a4a !important;
    border-radius: 12px;
}

/* ---------- Judul ---------- */
.stApp h1 {
    color: #0f2a4a;
    font-weight: 800;
    letter-spacing: -0.5px;
    padding-bottom: 0;
    margin-bottom: 0;
}

.stApp h2, .stApp h3 {
    color: #0f2a4a;
    font-weight: 700;
}

.page-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin: 0 0 1.2rem 0;
}

.date-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    float: right;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 8px 18px;
    color: #334155;
    font-size: 0.9rem;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(15, 42, 74, 0.06);
}

.date-badge .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.2);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2a4a 0%, #0b1f3a 60%, #0a1a31 100%);
    border-right: none;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 1.6rem;
    font-weight: 800;
    padding-top: 0.5rem;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

/* Menu radio -> tampilan navigasi */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 6px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 11px 16px;
    border-radius: 12px;
    transition: all 0.15s ease;
    cursor: pointer;
    width: 100%;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.08);
}

/* sembunyikan bulatan radio */
[data-testid="stSidebar"] div[role="radiogroup"] label > *:not(:has(p)):not(input) {
    display: none !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label > *:has(p) {
    display: block !important;
    margin-left: 0 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-weight: 500;
    font-size: 0.95rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #0d9488, #14b8a6);
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.35);
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700;
}

[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
    display: none;
}

/* Tombol sidebar */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: rgba(255, 255, 255, 0.08);
    color: #e2e8f0;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.16);
    border-color: rgba(255, 255, 255, 0.3);
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
}

/* ---------- Kartu KPI (st.metric) ---------- */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(15, 42, 74, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(15, 42, 74, 0.1);
}

[data-testid="stMetricLabel"] p {
    color: #475569;
    font-weight: 600;
    font-size: 0.9rem;
}

[data-testid="stMetricValue"] {
    color: #0f2a4a;
    font-weight: 800;
    font-size: 1.6rem;
}

/* cegah angka terpotong jadi "99...." */
[data-testid="stMetricValue"] > div,
[data-testid="stMetricLabel"] > div {
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
}

/* Warna pastel bergantian per kolom */
div[data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"],
div[data-testid="column"]:nth-child(1) [data-testid="stMetric"] {
    background: linear-gradient(135deg, #e8f1fe, #f5f9ff);
    border-color: #cfe0fb;
}

div[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"],
div[data-testid="column"]:nth-child(2) [data-testid="stMetric"] {
    background: linear-gradient(135deg, #e3f8ee, #f3fcf8);
    border-color: #c3ecd8;
}

div[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"],
div[data-testid="column"]:nth-child(3) [data-testid="stMetric"] {
    background: linear-gradient(135deg, #fff1e0, #fff9f1);
    border-color: #fbdcb4;
}

div[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"],
div[data-testid="column"]:nth-child(4) [data-testid="stMetric"] {
    background: linear-gradient(135deg, #fee7e7, #fff4f4);
    border-color: #fbc9c9;
}

div[data-testid="stColumn"]:nth-child(5) [data-testid="stMetric"],
div[data-testid="column"]:nth-child(5) [data-testid="stMetric"] {
    background: linear-gradient(135deg, #efe9fe, #f8f5ff);
    border-color: #dccffb;
}

/* ---------- Chart & tabel dalam kartu ---------- */
[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 10px 10px 4px 10px;
    box-shadow: 0 2px 8px rgba(15, 42, 74, 0.05);
}

[data-testid="stDataFrame"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 6px;
    box-shadow: 0 2px 8px rgba(15, 42, 74, 0.05);
    overflow: hidden;
}

/* ---------- Form ---------- */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(15, 42, 74, 0.05);
}

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}

[data-testid="stFileUploader"] section {
    background: #ffffff;
    border: 2px dashed #99f6e4;
    border-radius: 14px;
}

/* ---------- Tombol utama (teal) ---------- */
.stApp .stButton > button,
.stApp .stFormSubmitButton > button {
    background: linear-gradient(90deg, #0d9488, #14b8a6);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    box-shadow: 0 3px 10px rgba(13, 148, 136, 0.25);
}

.stApp .stButton > button:hover,
.stApp .stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #0f766e, #0d9488);
    color: #ffffff;
}

/* ---------- Kartu mesin ---------- */
.machine-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 6px solid var(--accent);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(15, 42, 74, 0.05);
}

.machine-card .mid {
    font-weight: 700;
    color: #0f2a4a;
    font-size: 1.05rem;
}

.machine-card .mname {
    color: #64748b;
    font-size: 0.8rem;
}

.machine-card .meta {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 6px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    margin-top: 6px;
}

.pill::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}

/* ---------- Alert list ---------- */
.alert-row {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 14px;
}

/* Sembunyikan footer bawaan */
footer { visibility: hidden; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ===============================================================
# HELPER
# ===============================================================

def style_fig(fig, height=380):
    """Terapkan tampilan chart yang konsisten dengan tema."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=13, color="#334155"),
        title=dict(font=dict(size=17, color="#0f2a4a")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=height,
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=""
        ),
        hoverlabel=dict(font_family="Inter")
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


def page_header(title, subtitle="Monitoring Kualitas Produksi Secara Real-Time"):
    now = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%d %b %Y, %H:%M WIB")

    st.markdown(
        f'<div class="date-badge"><span>📅 {now}</span>'
        f'<span class="dot"></span></div>',
        unsafe_allow_html=True
    )

    st.title(title)

    st.markdown(
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True
    )


def api_get(path, **params):
    try:
        r = requests.get(
            f"{API_URL}{path}",
            params=params,
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal terhubung ke API ({API_URL}): {e}")
        return None


def api_send(method, path, **kwargs):
    try:
        r = requests.request(
            method,
            f"{API_URL}{path}",
            timeout=60,
            **kwargs
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Request gagal: {e}")
        return None


# ===============================================================
# SIDEBAR
# ===============================================================

st.sidebar.title("🏭 SmartQC-Flow")
st.sidebar.caption("Deteksi anomali mesin & prediksi cacat produk")

page = st.sidebar.radio(
    "Menu",
    [
        "Overview",
        "Status Mesin",
        "Grafik Sensor",
        "Input Data",
        "Upload CSV",
        "Alerts",
        "Performa Model"
    ]
)

if st.sidebar.button("🔄 Refresh"):
    st.rerun()

health = api_get("/health")

if health:
    st.sidebar.success(
        f"API: {health['api']} | MongoDB: {health['mongodb']}"
    )


# ===============================================================
# OVERVIEW
# ===============================================================

if page == "Overview":

    page_header("Overview Lini Produksi")

    data = api_get("/stats/overview")

    if data:

        k = data["kpi"]

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Total Data Sensor", f"{k['total_logs']:,}")
        c2.metric("Total Prediksi", f"{k['total_predictions']:,}")
        c3.metric("Prediksi Cacat", f"{k['defective_predicted']:,}")
        c4.metric("Defect Rate", f"{k['defect_rate']:.1%}")
        c5.metric("Alert Aktif", k["active_alerts"])

        st.write("")

        col1, col2 = st.columns(2)

        dist = data["status_distribution"]

        if dist:
            dfd = pd.DataFrame({
                "status": list(dist.keys()),
                "jumlah": list(dist.values())
            })

            fig = px.pie(
                dfd,
                names="status",
                values="jumlah",
                hole=0.62,
                color="status",
                color_discrete_map=STATUS_COLOR,
                title="Distribusi Status Prediksi"
            )

            fig.update_traces(
                textinfo="percent",
                marker=dict(line=dict(color="#ffffff", width=3))
            )

            fig.add_annotation(
                text=f"<b>{int(dfd['jumlah'].sum()):,}</b><br>Data",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color="#0f2a4a")
            )

            style_fig(fig)

            col1.plotly_chart(fig, use_container_width=True, theme=None)

        hourly = api_get("/stats/defects-per-hour", hours=720)

        if hourly:
            dfh = pd.DataFrame(hourly)

            fig = px.bar(
                dfh,
                x="hour",
                y=["total", "defective"],
                barmode="group",
                title="Prediksi per Jam (Total vs Cacat)",
                color_discrete_map={
                    "total": "#3b82f6",
                    "defective": "#ef4444"
                }
            )

            fig.update_traces(marker_line_width=0)

            style_fig(fig)

            col2.plotly_chart(fig, use_container_width=True, theme=None)

        else:
            col2.info(
                "Belum ada prediksi. Kirim data lewat menu Input Data / Upload CSV."
            )


# ===============================================================
# STATUS MESIN
# ===============================================================

elif page == "Status Mesin":

    page_header("Status Mesin")

    machines = api_get("/machines")

    if machines:

        cols = st.columns(5)

        for i, m in enumerate(machines):

            color = STATUS_COLOR.get(
                m["status"],
                "#94a3b8"
            )

            pill_bg, pill_fg = STATUS_PILL.get(
                m["status"],
                ("#e2e8f0", "#475569")
            )

            prob = (
                f"{m['last_probability']:.0%}"
                if m["last_probability"] is not None
                else "-"
            )

            cols[i % 5].markdown(
                f"""
                <div class="machine-card" style="--accent:{color}">
                    <div class="mid">{m['machine_id']}</div>
                    <div class="mname">{m.get('name', '')}</div>
                    <span class="pill"
                          style="background:{pill_bg};color:{pill_fg}">
                        {m['status']}
                    </span>
                    <div class="meta">
                        Risiko: {prob} · Alert: {m['active_alerts']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.dataframe(
            pd.DataFrame(machines).drop(
                columns=["last_update"],
                errors="ignore"
            ),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Belum ada mesin. Jalankan import data (Anggota 2).")


# ===============================================================
# GRAFIK SENSOR
# ===============================================================

elif page == "Grafik Sensor":

    page_header("Grafik Sensor")

    machines = api_get("/machines") or []

    ids = [m["machine_id"] for m in machines]

    c1, c2 = st.columns([1, 1])

    mid = c1.selectbox("Mesin", ["Semua"] + ids)
    limit = c2.slider("Jumlah data terakhir", 20, 500, 150)

    logs = api_get(
        "/readings",
        machine_id=None if mid == "Semua" else mid,
        limit=limit
    )

    if logs:

        df = pd.DataFrame(logs).sort_values("timestamp")

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            format="mixed",
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"])

        for col, title in [
            ("process_temp", "Suhu Proses (K)"),
            ("rpm", "Kecepatan Putar (rpm)"),
            ("torque", "Torsi (Nm)"),
            ("tool_wear", "Tool Wear (menit)")
        ]:

            fig = px.line(
                df,
                x="timestamp",
                y=col,
                title=title,
                color="machine_id" if mid == "Semua" else None,
                color_discrete_sequence=px.colors.qualitative.Set2
                if mid == "Semua" else ["#3b82f6"]
            )

            fig.update_traces(line=dict(width=2.5))

            style_fig(fig, height=340)

            st.plotly_chart(
                fig,
                use_container_width=True, theme=None
            )

        preds = api_get(
            "/predictions",
            machine_id=None if mid == "Semua" else mid,
            limit=limit
        )

        if preds:

            dp = pd.DataFrame(preds).sort_values("timestamp")

            dp["timestamp"] = pd.to_datetime(
                dp["timestamp"],
                format="mixed",
                errors="coerce"
            )

            dp = dp.dropna(subset=["timestamp"])

            fig = px.scatter(
                dp,
                x="timestamp",
                y="defect_probability",
                color="status",
                color_discrete_map=STATUS_COLOR,
                title="Probabilitas Cacat"
            )

            fig.update_traces(
                marker=dict(size=8, line=dict(width=1, color="#ffffff"))
            )

            fig.add_hline(
                y=0.7,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="Ambang alert 70%"
            )

            style_fig(fig)

            st.plotly_chart(
                fig,
                use_container_width=True, theme=None
            )

    else:
        st.info("Belum ada data sensor.")


# ===============================================================
# INPUT DATA
# ===============================================================

elif page == "Input Data":

    page_header("Input Data Sensor Manual")

    with st.form("input_form"):

        c1, c2, c3 = st.columns(3)

        machine_id = c1.text_input("ID Mesin", "M-01")
        ptype = c2.selectbox("Tipe Produk (Kualitas)", ["L", "M", "H"])
        product_id = c3.text_input("ID Produk (opsional)", "")

        c1, c2 = st.columns(2)

        air = c1.number_input(
            "Suhu udara (K)", 280.0, 330.0, 300.0, 0.1
        )

        proc = c2.number_input(
            "Suhu proses (K)", 290.0, 350.0, 310.0, 0.1
        )

        c1, c2, c3 = st.columns(3)

        rpm = c1.number_input(
            "Kecepatan putar (rpm)", 0, 4000, 1500
        )

        torque = c2.number_input(
            "Torsi (Nm)", 0.0, 100.0, 40.0, 0.1
        )

        wear = c3.number_input(
            "Tool wear (menit)", 0, 400, 100
        )

        go = st.form_submit_button("Prediksi & Simpan")

    if go:

        payload = {
            "machine_id": machine_id,
            "type": ptype,
            "air_temp": air,
            "process_temp": proc,
            "rpm": rpm,
            "torque": torque,
            "tool_wear": wear
        }

        if product_id:
            payload["product_id"] = product_id

        res = api_send(
            "POST",
            "/readings",
            json=payload
        )

        if res:

            color = STATUS_COLOR[res["status"]]

            st.markdown(
                f"### Status: <span style='color:{color}'>{res['status']}</span>",
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Probabilitas cacat/gagal",
                f"{res['defect_probability']:.1%}"
            )

            c2.metric(
                "Skor anomali",
                f"{res['anomaly_score']:.2f}"
            )

            c3.metric(
                "Anomali?",
                "Ya" if res["is_anomaly"] else "Tidak"
            )

            if res.get("alert_id"):
                st.error(
                    "🚨 Alert dibuat karena risiko di atas ambang batas!"
                )

    with st.expander("Contoh data bermasalah"):
        st.code(
            "Tipe L | Suhu udara 300 | Suhu proses 308 | "
            "rpm 1300 | Torsi 65 | Tool wear 230"
        )


# ===============================================================
# UPLOAD CSV
# ===============================================================

elif page == "Upload CSV":

    page_header("Upload CSV Data Produksi")

    st.caption(
        "Format kolom AI4I: Type, Air temperature [K], "
        "Process temperature [K], Rotational speed [rpm], "
        "Torque [Nm], Tool wear [min]"
    )

    f = st.file_uploader(
        "Pilih file CSV",
        type=["csv"]
    )

    n = st.number_input(
        "Maksimum baris diproses",
        10,
        2000,
        300
    )

    if f and st.button("Proses"):

        with st.spinner("Memproses..."):

            res = api_send(
                "POST",
                f"/upload-csv?max_rows={int(n)}",
                files={
                    "file": (
                        f.name,
                        f.getvalue(),
                        "text/csv"
                    )
                }
            )

        if res:

            st.success(
                f"{res['rows_processed']} baris diproses, "
                f"{res['alerts_created']} alert dibuat"
            )

            st.json(res["status_summary"])


# ===============================================================
# ALERTS
# ===============================================================

elif page == "Alerts":

    page_header("Daftar Alert")

    only_active = st.checkbox(
        "Hanya yang belum selesai",
        True
    )

    alerts = (
        api_get("/alerts", resolved=False)
        if only_active
        else api_get("/alerts")
    )

    if alerts:

        for a in alerts[:50]:

            c1, c2 = st.columns([5, 1])

            icon = "✅" if a["resolved"] else "🚨"

            c1.markdown(
                f"<div class='alert-row'>{icon} <b>{a['machine_id']}</b> — "
                f"{a['message']}<br>"
                f"<small style='color:#64748b'>{a['timestamp']}</small></div>",
                unsafe_allow_html=True
            )

            if not a["resolved"] and c2.button(
                "Selesai",
                key=a["_id"]
            ):

                api_send(
                    "PATCH",
                    f"/alerts/{a['_id']}/resolve"
                )

                st.rerun()

    else:
        st.success("Tidak ada alert.")


# ===============================================================
# PERFORMA MODEL
# ===============================================================

elif page == "Performa Model":

    page_header("Performa Model Machine Learning")

    m = api_get("/model/metrics")

    if m:

        model_name = m.get("model", "Gradient Boosting")

        st.markdown(
            f"""
            <div style="margin-bottom:16px">
                <span class="pill" style="background:#ccfbf1;color:#0f766e;
                      font-size:0.9rem;padding:6px 16px">
                    Model: {model_name}
                </span>
                <span class="pill" style="background:#e0e7ff;color:#3730a3;
                      font-size:0.9rem;padding:6px 16px;margin-left:8px">
                    Deteksi anomali: Isolation Forest
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision']:.2%}")
        c3.metric("Recall", f"{m['recall']:.2%}")
        c4.metric("F1", f"{m['f1']:.2%}")
        c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

        st.write("")

        st.write(
            "Confusion matrix (baris=aktual, kolom=prediksi):"
        )

        st.dataframe(
            pd.DataFrame(
                m["confusion_matrix"],
                index=["Aktual Normal", "Aktual Gagal"],
                columns=["Pred Normal", "Pred Gagal"]
            )
        )

        st.write(
            "Isolation Forest:",
            m["isolation_forest"]
        )

        st.caption(
            f"Dataset: {m['dataset']} | "
            f"train={m['n_train']} | test={m['n_test']}"
        )