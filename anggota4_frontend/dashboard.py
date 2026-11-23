"""
SMARTQC-FLOW - Dashboard Monitoring Kualitas Produksi
Jalankan dari ROOT project:
    streamlit run anggota4_frontend/dashboard.py
"""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# =========================================================
# KONFIGURASI
# =========================================================
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "anggota3_ml" / "models"

st.set_page_config(
    page_title="SmartQC-Flow | Monitoring Produksi",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_COLOR = {
    "Normal": "#238B67",
    "Waspada": "#E99A28",
    "Bahaya": "#E34D59",
    "Belum ada data": "#8793A5",
}

NAV_ITEMS = [
    "Dashboard",
    "Data Mesin",
    "Grafik Sensor",
    "Input Data",
    "Upload CSV",
    "Alert",
    "Data Historis",
    "Evaluasi Model",
]

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F3F6FC;
        color: #14213D;
    }

    [data-testid="stHeader"] {
        background: rgba(243,246,252,0.92);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102344 0%, #142D52 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #EAF1FC;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #B9C8DE;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border-radius: 11px;
        padding: 10px 12px;
        margin: 3px 0;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #078F91;
        color: white;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-weight: 600;
        color: inherit;
    }

    .brand {
        padding: 12px 8px 24px 8px;
        text-align: center;
    }

    .brand-icon {
        font-size: 42px;
        line-height: 1.3;
    }

    .brand-name {
        color: white;
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand-caption {
        color: #AFC2DD;
        font-size: 11px;
        margin-top: 4px;
    }

    .page-title {
        color: #14213D;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }

    .page-subtitle {
        color: #65748B;
        font-size: 14px;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    .top-meta {
        color: #607089;
        font-size: 12px;
        text-align: right;
        padding-top: 10px;
    }

    .kpi-card {
        min-height: 126px;
        border: 1px solid #DCE5F2;
        border-radius: 16px;
        padding: 17px 18px 13px 18px;
        background: white;
        box-shadow: 0 3px 12px rgba(22, 45, 82, 0.045);
    }

    .kpi-label {
        color: #52627A;
        font-size: 13px;
        font-weight: 700;
    }

    .kpi-value {
        color: #14213D;
        font-size: 30px;
        font-weight: 800;
        margin-top: 7px;
        line-height: 1.1;
    }

    .kpi-note {
        color: #75839A;
        font-size: 11px;
        margin-top: 8px;
    }

    .panel {
        background: white;
        border: 1px solid #DFE7F2;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 3px 12px rgba(22, 45, 82, 0.04);
        margin-bottom: 14px;
    }

    .panel-title {
        color: #1B2B49;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .model-strip {
        background: linear-gradient(100deg, #E6F5F3, #EDF4FF);
        border: 1px solid #CDE8E5;
        border-radius: 14px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
    }

    .model-name {
        color: #087F80;
        font-size: 18px;
        font-weight: 800;
    }

    .model-detail {
        color: #586A80;
        font-size: 12px;
        margin-top: 3px;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E0E7F1;
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 0;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: #078F91;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUNGSI API DAN KOMPONEN TAMPILAN
# =========================================================
def api_get(path, show_error=True, **params):
    """Mengambil data dari backend FastAPI."""
    try:
        response = requests.get(
            f"{API_URL}{path}",
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        if show_error:
            st.error(f"Gagal terhubung ke API ({API_URL}): {exc}")
        return None


def api_send(method, path, show_error=True, **kwargs):
    """Mengirim data atau perintah ke backend FastAPI."""
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            timeout=60,
            **kwargs,
        )
        response.raise_for_status()

        if response.content:
            return response.json()

        return {}

    except requests.RequestException as exc:
        if show_error:
            st.error(f"Request ke API gagal: {exc}")
        return None


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(container, label, value, note, accent="#168EAA"):
    container.markdown(
        f"""
        <div class="kpi-card" style="border-top:4px solid {accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_style(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#53627A",
            size=11,
        ),
        title=dict(font=dict(size=15, color="#1B2B49")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(showgrid=False, linecolor="#E5EAF2")
    fig.update_yaxes(gridcolor="#E9EEF5", zeroline=False)

    return fig


def status_color(status):
    return STATUS_COLOR.get(
        status,
        STATUS_COLOR["Belum ada data"],
    )


def show_model_strip(metrics=None):
    accuracy = (
        f" | Accuracy pengujian: {metrics['accuracy']:.2%}"
        if metrics and metrics.get("accuracy") is not None
        else ""
    )

    st.markdown(
        f"""
        <div class="model-strip">
            <div class="model-name">
                ⚙️ Model Utama: Gradient Boosting
            </div>
            <div class="model-detail">
                Model klasifikasi untuk memperkirakan risiko cacat produk
                berdasarkan data sensor mesin.{accuracy}
                Isolation Forest digunakan sebagai deteksi anomali tambahan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_empty(message):
    st.info(message)


# =========================================================
# SIDEBAR / NAVIGASI
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🏭</div>
            <div class="brand-name">SmartQC-Flow</div>
            <div class="brand-caption">Industrial Quality Monitoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGASI",
        NAV_ITEMS,
        format_func=lambda x: {
            "Dashboard": "⌂  Dashboard",
            "Data Mesin": "⚙  Data Mesin",
            "Grafik Sensor": "▥  Grafik Sensor",
            "Input Data": "＋  Input Data",
            "Upload CSV": "⇧  Upload CSV",
            "Alert": "♢  Alert",
            "Data Historis": "◷  Data Historis",
            "Evaluasi Model": "▤  Evaluasi Model",
        }[x],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if st.button("↻  Refresh Data", use_container_width=True):
        st.rerun()

    health = api_get("/health", show_error=False)

    if health:
        st.success(
            f"● Sistem Online\n\n"
            f"API: {health.get('api', '-')}\n\n"
            f"MongoDB: {health.get('mongodb', '-')}"
        )
    else:
        st.warning("● API belum terhubung")

    st.caption("SmartQC-Flow · Machine Learning Monitoring")


# =========================================================
# DASHBOARD UTAMA
# =========================================================
if page == "Dashboard":
    header_left, header_right = st.columns([4, 1])

    with header_left:
        page_header(
            "Dashboard",
            "Monitoring Kualitas Produksi Secara Real-Time",
        )

    with header_right:
        st.markdown(
            '<div class="top-meta">● Live Monitoring</div>',
            unsafe_allow_html=True,
        )

    overview = api_get("/stats/overview")
    machines = api_get("/machines", show_error=False) or []
    metrics = api_get("/model/metrics", show_error=False)

    if overview:
        kpi = overview.get("kpi", {})

        total_logs = kpi.get("total_logs", 0)
        total_predictions = kpi.get("total_predictions", 0)
        defective = kpi.get("defective_predicted", 0)
        active_alerts = kpi.get("active_alerts", 0)
        defect_rate = kpi.get("defect_rate", 0)

        cards = st.columns(4, gap="medium")

        kpi_card(
            cards[0],
            "Total Data Sensor",
            f"{total_logs:,}",
            "Seluruh data sensor tersimpan",
            "#2C83D5",
        )

        kpi_card(
            cards[1],
            "Total Prediksi",
            f"{total_predictions:,}",
            "Prediksi yang telah diproses",
            "#13A58B",
        )

        kpi_card(
            cards[2],
            "Prediksi Cacat",
            f"{defective:,}",
            f"Defect rate: {defect_rate:.2%}",
            "#E99A28",
        )

        kpi_card(
            cards[3],
            "Alert Aktif",
            f"{active_alerts:,}",
            "Alert yang belum diselesaikan",
            "#E34D59",
        )

    else:
        st.warning(
            "Ringkasan belum tersedia. Pastikan backend dan MongoDB aktif."
        )

    show_model_strip(metrics)

    left, right = st.columns([1.55, 1], gap="medium")

    with left:
        st.markdown(
            '<div class="panel-title">▥ Tren Prediksi Produksi</div>',
            unsafe_allow_html=True,
        )

        hourly = api_get(
            "/stats/defects-per-hour",
            hours=24,
            show_error=False,
        )

        if hourly:
            df_hour = pd.DataFrame(hourly)

            if {"hour", "total", "defective"}.issubset(df_hour.columns):
                df_hour = df_hour.sort_values("hour")

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df_hour["hour"],
                        y=df_hour["total"],
                        name="Total Prediksi",
                        mode="lines+markers",
                        line=dict(color="#347FE0", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(52,127,224,0.08)",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_hour["hour"],
                        y=df_hour["defective"],
                        name="Prediksi Cacat",
                        mode="lines+markers",
                        line=dict(color="#E34D59", width=3),
                    )
                )

                fig.update_layout(
                    title="Jumlah prediksi per jam (24 jam terakhir)",
                    xaxis_title="Waktu",
                    yaxis_title="Jumlah",
                )

                st.plotly_chart(
                    apply_chart_style(fig, 330),
                    use_container_width=True,
                )

            else:
                show_empty("Format data tren dari API belum sesuai.")

        else:
            show_empty("Belum ada data prediksi per jam.")

    with right:
        st.markdown(
            '<div class="panel-title">◉ Status Prediksi</div>',
            unsafe_allow_html=True,
        )

        distribution = (
            overview.get("status_distribution", {})
            if overview else {}
        )

        if distribution and sum(distribution.values()) > 0:
            df_dist = pd.DataFrame({
                "Status": list(distribution.keys()),
                "Jumlah": list(distribution.values()),
            })

            fig = px.pie(
                df_dist,
                names="Status",
                values="Jumlah",
                hole=0.62,
                color="Status",
                color_discrete_map=STATUS_COLOR,
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                marker=dict(line=dict(color="white", width=3)),
            )

            fig.add_annotation(
                text=f"<b>{sum(distribution.values()):,}</b><br>Data",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=19, color="#1B2B49"),
            )

            st.plotly_chart(
                apply_chart_style(fig, 330),
                use_container_width=True,
            )

        else:
            show_empty("Belum ada distribusi status prediksi.")

    st.markdown(
        '<div class="panel-title">⚙ Mesin Terpantau</div>',
        unsafe_allow_html=True,
    )

    if machines:
        machine_df = pd.DataFrame(machines)

        preferred = [
            "machine_id",
            "name",
            "status",
            "last_probability",
            "active_alerts",
            "last_update",
        ]

        visible = [
            col for col in preferred
            if col in machine_df.columns
        ]

        machine_df = machine_df[visible].copy()

        rename = {
            "machine_id": "Machine ID",
            "name": "Nama Mesin",
            "status": "Status",
            "last_probability": "Probabilitas Risiko",
            "active_alerts": "Alert Aktif",
            "last_update": "Pembaruan Terakhir",
        }

        machine_df = machine_df.rename(columns=rename)

        if "Probabilitas Risiko" in machine_df:
            machine_df["Probabilitas Risiko"] = machine_df[
                "Probabilitas Risiko"
            ].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "-"
            )

        st.dataframe(
            machine_df.head(8),
            use_container_width=True,
            hide_index=True,
        )

    else:
        show_empty(
            "Belum ada data mesin. Jalankan import data dari Anggota 2."
        )

        # =========================================================
# DATA MESIN
# =========================================================
elif page == "Data Mesin":
    page_header(
        "Data Mesin",
        "Daftar mesin dan kondisi prediksi terakhir",
    )

    machines = api_get("/machines")

    if machines:
        cols = st.columns(3)

        for index, machine in enumerate(machines):
            status = machine.get("status", "Belum ada data")
            color = status_color(status)
            probability = machine.get("last_probability")

            probability_text = (
                f"{probability:.1%}"
                if probability is not None
                else "-"
            )

            with cols[index % 3]:
                st.markdown(
                    f"""
                    <div class="kpi-card"
                         style="border-left:5px solid {color};
                         margin-bottom:14px">
                        <div class="kpi-label">
                            {machine.get('machine_id', '-')}
                        </div>
                        <div class="kpi-value" style="font-size:20px">
                            {machine.get('name') or 'Mesin Produksi'}
                        </div>
                        <div style="color:{color};font-weight:800;margin-top:8px">
                            ● {status}
                        </div>
                        <div class="kpi-note">
                            Risiko: {probability_text} ·
                            Alert aktif: {machine.get('active_alerts', 0)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("#### Detail Mesin")

        st.dataframe(
            pd.DataFrame(machines),
            use_container_width=True,
            hide_index=True,
        )

    else:
        show_empty("Belum ada mesin yang tercatat.")


# =========================================================
# GRAFIK SENSOR
# =========================================================
elif page == "Grafik Sensor":
    page_header(
        "Grafik Sensor",
        "Pemantauan perubahan parameter mesin",
    )

    machines = api_get("/machines", show_error=False) or []

    machine_ids = [
        m["machine_id"]
        for m in machines
        if m.get("machine_id")
    ]

    filter_col, limit_col = st.columns([1, 1])

    selected_machine = filter_col.selectbox(
        "Pilih Mesin",
        ["Semua"] + machine_ids,
    )

    limit = limit_col.slider(
        "Jumlah data terakhir",
        20,
        500,
        150,
    )

    readings = api_get(
        "/readings",
        machine_id=(
            None if selected_machine == "Semua"
            else selected_machine
        ),
        limit=limit,
    )

    if readings:
        df = pd.DataFrame(readings)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

            df = df.sort_values("timestamp")

        sensor_fields = [
            ("process_temp", "Suhu Proses (K)"),
            ("air_temp", "Suhu Udara (K)"),
            ("rpm", "Kecepatan Putar (rpm)"),
            ("torque", "Torsi (Nm)"),
            ("tool_wear", "Tool Wear (menit)"),
        ]

        for field, title in sensor_fields:
            if field in df.columns:
                fig = px.line(
                    df,
                    x="timestamp",
                    y=field,
                    color=(
                        "machine_id"
                        if selected_machine == "Semua"
                        and "machine_id" in df.columns
                        else None
                    ),
                    title=title,
                )

                st.plotly_chart(
                    apply_chart_style(fig, 280),
                    use_container_width=True,
                )

        predictions = api_get(
            "/predictions",
            machine_id=(
                None if selected_machine == "Semua"
                else selected_machine
            ),
            limit=limit,
            show_error=False,
        )

        if predictions:
            dp = pd.DataFrame(predictions)

            dp["timestamp"] = pd.to_datetime(
                dp["timestamp"],
                errors="coerce",
            )

            dp = dp.sort_values("timestamp")

            if {
                "timestamp",
                "defect_probability",
            }.issubset(dp.columns):

                fig = px.scatter(
                    dp,
                    x="timestamp",
                    y="defect_probability",
                    color="status",
                    color_discrete_map=STATUS_COLOR,
                    title="Probabilitas Cacat Produk",
                )

                fig.add_hline(
                    y=0.70,
                    line_dash="dash",
                    line_color="#E34D59",
                    annotation_text="Ambang alert 70%",
                )

                st.plotly_chart(
                    apply_chart_style(fig, 300),
                    use_container_width=True,
                )

    else:
        show_empty(
            "Belum ada data sensor untuk ditampilkan."
        )


# =========================================================
# INPUT DATA
# =========================================================
elif page == "Input Data":
    page_header(
        "Input Data Sensor",
        "Masukkan pembacaan sensor mesin untuk diprediksi",
    )

    with st.form("input_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        machine_id = col1.text_input(
            "ID Mesin",
            "M-01",
        )

        product_type = col2.selectbox(
            "Tipe Produk",
            ["L", "M", "H"],
        )

        product_id = col3.text_input(
            "ID Produk (opsional)",
            "",
        )

        col1, col2 = st.columns(2)

        air_temp = col1.number_input(
            "Suhu Udara (K)",
            280.0,
            330.0,
            300.0,
            0.1,
        )

        process_temp = col2.number_input(
            "Suhu Proses (K)",
            290.0,
            350.0,
            310.0,
            0.1,
        )

        col1, col2, col3 = st.columns(3)

        rpm = col1.number_input(
            "Kecepatan Putar (rpm)",
            0,
            4000,
            1500,
        )

        torque = col2.number_input(
            "Torsi (Nm)",
            0.0,
            100.0,
            40.0,
            0.1,
        )

        tool_wear = col3.number_input(
            "Tool Wear (menit)",
            0,
            400,
            100,
        )

        submitted = st.form_submit_button(
            "Prediksi dan Simpan Data",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "machine_id": machine_id,
            "type": product_type,
            "air_temp": air_temp,
            "process_temp": process_temp,
            "rpm": rpm,
            "torque": torque,
            "tool_wear": tool_wear,
        }

        if product_id.strip():
            payload["product_id"] = product_id.strip()

        result = api_send(
            "POST",
            "/readings",
            json=payload,
        )

        if result:
            status = result.get(
                "status",
                "Belum ada data",
            )

            color = status_color(status)

            st.markdown(
                f"""
                ### Hasil Prediksi:
                <span style="color:{color}">{status}</span>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Probabilitas Cacat",
                f"{result.get('defect_probability', 0):.2%}",
            )

            c2.metric(
                "Skor Anomali",
                f"{result.get('anomaly_score', 0):.4f}",
            )

            c3.metric(
                "Terdeteksi Anomali",
                "Ya" if result.get("is_anomaly") else "Tidak",
            )

            if result.get("alert_id"):
                st.error(
                    "Alert dibuat karena hasil prediksi "
                    "memenuhi kondisi alert."
                )

    with st.expander("Contoh data sensor bermasalah"):
        st.code(
            "Tipe L | Suhu udara 300 K | Suhu proses 308 K | "
            "RPM 1300 | Torsi 65 Nm | Tool wear 230 menit"
        )


# =========================================================
# UPLOAD CSV
# =========================================================
elif page == "Upload CSV":
    page_header(
        "Upload CSV",
        "Proses data produksi secara batch",
    )

    st.info(
        "Format dataset AI4I: Type, Air temperature [K], "
        "Process temperature [K], Rotational speed [rpm], "
        "Torque [Nm], dan Tool wear [min]."
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
    )

    max_rows = st.number_input(
        "Maksimum baris yang diproses",
        10,
        2000,
        300,
        step=10,
    )

    if uploaded_file and st.button(
        "Proses Data CSV",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Mengirim dan memproses data melalui backend..."
        ):
            result = api_send(
                "POST",
                f"/upload-csv?max_rows={int(max_rows)}",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv",
                    )
                },
            )

        if result:
            st.success(
                f"{result.get('rows_processed', 0)} baris diproses · "
                f"{result.get('alerts_created', 0)} alert dibuat"
            )

            summary = result.get("status_summary", {})

            if summary:
                cols = st.columns(3)

                for index, status in enumerate(
                    ["Normal", "Waspada", "Bahaya"]
                ):
                    cols[index].metric(
                        status,
                        summary.get(status, 0),
                    )

                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Status": key, "Jumlah": value}
                            for key, value in summary.items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )



# ===============================================================
# ALERTS
elif page == "Alert":
    st.title("Daftar Alert")
    st.caption("Kelola dan pantau peringatan kondisi mesin.")

    only_active = st.checkbox("Hanya yang belum selesai", True)

    # Ambil data alert dari backend
    if only_active:
        alerts = api_get("/alerts", resolved=False)
    else:
        alerts = api_get("/alerts")

    if alerts is not None and len(alerts) > 0:

        for a in alerts[:50]:
            c1, c2 = st.columns([5, 1])

            alert_id = str(a["_id"])
            is_resolved = a.get("resolved", False)

            icon = "✅" if is_resolved else "🚨"

            with c1:
                st.markdown(
                    f"{icon} **{a['machine_id']}** — {a['message']}"
                )
                st.caption(a.get("timestamp", "-"))

                if is_resolved:
                    st.success("Alert sudah diselesaikan.")
                else:
                    st.warning("Alert belum diselesaikan.")

            with c2:
                if not is_resolved:
                    if st.button(
                        "Selesaikan",
                        key=f"resolve_{alert_id}"
                    ):
                        with st.spinner("Memperbarui status alert..."):

                            # Kirim permintaan ke backend
                            result = api_send(
                                "PATCH",
                                f"/alerts/{alert_id}/resolve"
                            )

                            # Ambil ulang data alert untuk verifikasi
                            updated_alerts = api_get("/alerts")

                            updated_alert = None

                            if updated_alerts is not None:
                                updated_alert = next(
                                    (
                                        item for item in updated_alerts
                                        if str(item.get("_id")) == alert_id
                                    ),
                                    None
                                )

                            # Pastikan status benar-benar berubah
                            if (
                                updated_alert is not None
                                and updated_alert.get("resolved") is True
                            ):
                                st.toast(
                                    "Alert berhasil diselesaikan!",
                                    icon="✅"
                                )

                                st.rerun()

                            else:
                                st.error(
                                    "Status alert belum berubah. "
                                    "Periksa respons backend dan koneksi database."
                                )

            st.divider()

    elif alerts is not None:
        st.success("Tidak ada alert.")

    else:
        st.error("Gagal mengambil data alert dari backend.")
# =========================================================
# DATA HISTORIS
# =========================================================
elif page == "Data Historis":
    page_header(
        "Data Historis",
        "Riwayat pembacaan sensor dan hasil prediksi",
    )

    machines = api_get(
        "/machines",
        show_error=False,
    ) or []

    machine_ids = [
        m["machine_id"]
        for m in machines
        if m.get("machine_id")
    ]

    selected_machine = st.selectbox(
        "Filter Mesin",
        ["Semua"] + machine_ids,
        key="history_machine",
    )

    limit = st.slider(
        "Jumlah riwayat",
        20,
        500,
        100,
        key="history_limit",
    )

    logs = api_get(
        "/readings",
        machine_id=(
            None if selected_machine == "Semua"
            else selected_machine
        ),
        limit=limit,
    )

    if logs:
        df = pd.DataFrame(logs)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

            df = df.sort_values(
                "timestamp",
                ascending=False,
            )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Unduh Riwayat sebagai CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="smartqc_riwayat_sensor.csv",
            mime="text/csv",
        )

    else:
        show_empty(
            "Belum ada riwayat pembacaan sensor."
        )


# =========================================================
# EVALUASI MODEL MACHINE LEARNING
# =========================================================
elif page == "Evaluasi Model":
    page_header(
        "Evaluasi Model Machine Learning",
        "Kinerja model klasifikasi Gradient Boosting "
        "dan deteksi anomali",
    )

    metrics = api_get("/model/metrics")

    if metrics:
        show_model_strip(metrics)

        st.markdown(
            "#### Metrik Evaluasi Gradient Boosting"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accuracy",
            f"{metrics.get('accuracy', 0):.2%}",
        )

        c2.metric(
            "Precision",
            f"{metrics.get('precision', 0):.2%}",
        )

        c3.metric(
            "Recall",
            f"{metrics.get('recall', 0):.2%}",
        )

        c4.metric(
            "F1-Score",
            f"{metrics.get('f1', 0):.2%}",
        )

        c5.metric(
            "ROC-AUC",
            f"{metrics.get('roc_auc', 0):.3f}",
        )

        st.caption(
            f"Dataset: {metrics.get('dataset', '-')} · "
            f"Data training: {metrics.get('n_train', '-')} · "
            f"Data testing: {metrics.get('n_test', '-')}"
        )

        # Threshold produksi dan threshold analisis
        # memiliki tujuan yang berbeda.
        threshold_col1, threshold_col2 = st.columns(2)

        threshold_col1.metric(
            "Threshold Alert Produksi",
            "70%",
            help=(
                "Nilai default ALERT_THRESHOLD "
                "yang digunakan predict.py."
            ),
        )

        threshold_col2.metric(
            "Threshold Terbaik Analisis",
            "59,58%",
            help=(
                "Hasil pencarian threshold berdasarkan F1 "
                "pada prediksi out-of-fold data training."
            ),
        )

        st.caption(
            "Kedua threshold memiliki tujuan berbeda. "
            "Threshold alert produksi saat ini tetap 70%; "
            "hasil analisis 59,58% merupakan threshold terbaik "
            "pada evaluasi training dan tidak otomatis "
            "mengubah konfigurasi prediksi."
        )

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------
        st.markdown("#### Confusion Matrix")

        confusion = metrics.get("confusion_matrix")

        if confusion:
            matrix_df = pd.DataFrame(
                confusion,
                index=[
                    "Aktual Normal",
                    "Aktual Gagal",
                ],
                columns=[
                    "Prediksi Normal",
                    "Prediksi Gagal",
                ],
            )

            st.dataframe(
                matrix_df,
                use_container_width=True,
            )

        # -------------------------------------------------
        # GAMBAR EVALUASI MODEL
        # -------------------------------------------------
        cm_image = MODEL_DIR / "confusion_matrix.png"
        fi_image = MODEL_DIR / "feature_importance.png"

        image_col1, image_col2 = st.columns(2)

        if cm_image.exists():
            image_col1.markdown(
                "#### Visualisasi Confusion Matrix"
            )

            image_col1.image(
                str(cm_image),
                use_container_width=True,
            )

        if fi_image.exists():
            image_col2.markdown(
                "#### Feature Importance Gradient Boosting"
            )

            image_col2.image(
                str(fi_image),
                use_container_width=True,
            )

        # -------------------------------------------------
        # ISOLATION FOREST
        # -------------------------------------------------
        st.markdown(
            "#### Deteksi Anomali Tambahan — Isolation Forest"
        )

        isolation = metrics.get(
            "isolation_forest",
            {},
        )

        if isolation:
            iso_cols = st.columns(3)

            iso_cols[0].metric(
                "Anomaly Rate Test",
                f"{isolation.get('anomaly_rate_test', 0):.2%}",
            )

            iso_cols[1].metric(
                "Recall vs Failure",
                f"{isolation.get('recall_vs_failure', 0):.2%}",
            )

            iso_cols[2].metric(
                "Precision vs Failure",
                f"{isolation.get('precision_vs_failure', 0):.2%}",
            )

            with st.expander(
                "Detail skor Isolation Forest"
            ):
                st.json(isolation)

        else:
            st.info(
                "Informasi Isolation Forest belum tersedia "
                "di metrics.json."
            )

        # -------------------------------------------------
        # CATATAN EVALUASI
        # -------------------------------------------------
        st.warning(
            "Catatan evaluasi: recall menunjukkan proporsi "
            "data gagal yang berhasil ditemukan model. "
            "Karena recall pengujian belum 100%, sebagian "
            "kasus gagal masih dapat terlewat. Hasil evaluasi "
            "bukan jaminan bahwa seluruh kerusakan akan terdeteksi."
        )

    else:
        st.error(
            "Metrik model belum dapat dimuat. Pastikan model "
            "sudah dilatih dan endpoint GET /model/metrics "
            "dapat diakses."
        )