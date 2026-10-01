import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from influxdb_client_3 import InfluxDBClient3


# ============================================================
# 1. CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Electricity Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CSS GIAO DIỆN
#
# Phong cách:
# - hiện đại
# - tối giản
# - không dùng emoji / icon trang trí
# - navigation ngang
# - card KPI
# - biểu đồ có khung
# - hiệu ứng chuyển nội dung nhẹ
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background: linear-gradient(
        180deg,
        #f8fafc 0%,
        #eef2f7 100%
    );
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* ============================================================
   HEADER
   ============================================================ */

.dashboard-header {
    padding: 30px 34px;
    margin-bottom: 18px;
    border-radius: 20px;

    background: linear-gradient(
        135deg,
        #111827 0%,
        #1f2937 55%,
        #334155 100%
    );

    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.14);

    animation: fadeDown 0.45s ease-out;
}

.dashboard-kicker {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 8px;
}

.dashboard-title {
    font-size: 38px;
    line-height: 1.1;
    font-weight: 750;
    letter-spacing: -1.2px;
    color: #ffffff;
    margin-bottom: 8px;
}

.dashboard-subtitle {
    font-size: 15px;
    line-height: 1.6;
    color: #cbd5e1;
    margin: 0;
}


/* ============================================================
   NAVIGATION
   ============================================================ */

.navigation-box {
    margin-bottom: 20px;
}

[data-testid="stRadio"] {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 6px 10px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
}

[data-testid="stRadio"] > div {
    gap: 6px;
}

[data-testid="stRadio"] label {
    border-radius: 10px;
    padding: 8px 14px;
    transition:
        background 0.18s ease,
        color 0.18s ease,
        transform 0.18s ease;
}

[data-testid="stRadio"] label:hover {
    background: #f1f5f9;
    transform: translateY(-1px);
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] label {
    font-weight: 550;
}

/* About page: ẩn sidebar khi không cần bộ lọc */
body.about-page section[data-testid="stSidebar"] {
    display: none;
}

.sidebar-note {
    color: #64748b;
    font-size: 13px;
    line-height: 1.5;
}


/* ============================================================
   PAGE CONTENT
   ============================================================ */

.page-content {
    animation: pageFade 0.30s ease-out;
}

.page-eyebrow {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    color: #64748b;
    margin-bottom: 5px;
}

.page-title {
    font-size: 30px;
    font-weight: 740;
    letter-spacing: -0.8px;
    color: #0f172a;
    margin-bottom: 5px;
}

.page-description {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 24px;
}


/* ============================================================
   SECTION TITLE
   ============================================================ */

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #111827;
    margin-top: 24px;
    margin-bottom: 10px;
}

.section-caption {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 12px;
}


/* ============================================================
   KPI CARDS
   ============================================================ */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px 20px;

    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);

    transition:
        transform 0.20s ease,
        box-shadow 0.20s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.10);
}


/* ============================================================
   CHARTS
   ============================================================ */

div[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 8px;

    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);

    transition:
        transform 0.20s ease,
        box-shadow 0.20s ease;
}

div[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
}


/* ============================================================
   DATAFRAME / EXPANDER
   ============================================================ */

div[data-testid="stDataFrame"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;

    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
}

div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
}


/* ============================================================
   INFO BLOCKS
   ============================================================ */

.model-info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 122px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
}

.model-info-label {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 8px;
}

.model-info-value {
    font-size: 27px;
    line-height: 1.2;
    font-weight: 700;
    color: #0f172a;
}

.architecture-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 12px 0 8px 0;
}

.architecture-step {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 14px;
    min-height: 94px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.04);
    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease;
}

.architecture-step:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.architecture-number {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.architecture-name {
    font-size: 14px;
    font-weight: 700;
    color: #0f172a;
}

@media (max-width: 900px) {
    .architecture-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 600px) {
    .architecture-grid {
        grid-template-columns: 1fr;
    }
}


.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 22px;
    height: 100%;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
}

.info-card-title {
    font-weight: 700;
    font-size: 16px;
    color: #111827;
    margin-bottom: 8px;
}

.info-card-text {
    font-size: 14px;
    line-height: 1.65;
    color: #64748b;
}


/* ============================================================
   PIPELINE
   ============================================================ */

.pipeline {
    display: flex;
    align-items: stretch;
    gap: 10px;
    flex-wrap: wrap;
    margin: 10px 0 22px 0;
}

.pipeline-step {
    flex: 1 1 150px;
    min-width: 140px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.04);
}

.pipeline-number {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.pipeline-name {
    font-size: 14px;
    font-weight: 700;
    color: #0f172a;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    margin-top: 40px;
    padding: 20px 0 4px 0;
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    border-top: 1px solid #e2e8f0;
}


/* ============================================================
   ANIMATION
   ============================================================ */

@keyframes fadeDown {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pageFade {
    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. CẤU HÌNH INFLUXDB
# ============================================================

HOST = "http://127.0.0.1:8181"
DATABASE = "electricity"
MEASUREMENT = "electricity_forecast"

TOKEN = os.getenv("INFLUXDB3_AUTH_TOKEN")


# ============================================================
# 4. KIỂM TRA TOKEN
# ============================================================

if not TOKEN:
    st.error("Chưa tìm thấy INFLUXDB3_AUTH_TOKEN.")

    st.info("Hãy chạy lệnh sau trong PowerShell:")

    st.code(
        """
$env:INFLUXDB3_AUTH_TOKEN = (
    Get-Content "D:\\BigData_Electricity\\influxdb-token.txt" -Raw
).Trim()
        """,
        language="powershell",
    )

    st.stop()


# ============================================================
# 5. HÀM QUERY INFLUXDB
#
# Cache trong 60 giây để tránh query lặp lại liên tục.
# ============================================================

@st.cache_data(ttl=60)
def run_query(sql: str):
    client = InfluxDBClient3(
        host=HOST,
        database=DATABASE,
        token=TOKEN,
    )

    try:
        result = client.query(
            query=sql,
            language="sql",
            mode="pandas",
        )

        return result

    finally:
        client.close()


# ============================================================
# 6. LẤY DANH SÁCH SITE
# ============================================================

@st.cache_data(ttl=300)
def get_sites():
    sql = f"""
        SELECT DISTINCT site_id
        FROM "{MEASUREMENT}"
        WHERE time >= '2017-01-01T00:00:00Z'
          AND time < '2018-01-01T00:00:00Z'
        ORDER BY site_id
    """
    result = run_query(sql)
    if result.empty:
        return []
    return result["site_id"].astype(str).tolist()


sites = get_sites()


# ============================================================
# 7. KIỂM TRA DỮ LIỆU
# ============================================================

if not sites:
    st.error(
        "Không tìm thấy dữ liệu electricity_forecast trong InfluxDB."
    )
    st.stop()


# ============================================================
# 8. HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-kicker">Big Data · Time Series · Machine Learning</div>
        <div class="dashboard-title">Electricity Analytics</div>
        <div class="dashboard-subtitle">
            Phân tích tiêu thụ điện năng và kết quả dự báo trên dữ liệu năm 2017
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    [
        "Overview",
        "Analysis",
        "Forecast",
        "Anomaly",
        "About",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# Đánh dấu trang About để CSS có thể ẩn sidebar.
if page == "About":
    st.markdown(
        "<style>body { } section[data-testid='stSidebar'] { display: none; }</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# 10. SIDEBAR FILTER
# ============================================================

st.sidebar.markdown("## Bộ lọc dữ liệu")

st.sidebar.markdown(
    """
    <div class="sidebar-note">
        Các biểu đồ và chỉ số trên các trang phân tích
        được truy vấn trực tiếp từ InfluxDB.
    </div>
    """,
    unsafe_allow_html=True,
)

selected_site = st.sidebar.selectbox(
    "Site",
    ["Tất cả"] + sites,
)

date_range = st.sidebar.date_input(
    "Khoảng thời gian",
    value=(
        date(2017, 1, 1),
        date(2017, 12, 31),
    ),
    min_value=date(2017, 1, 1),
    max_value=date(2017, 12, 31),
)


# ============================================================
# 11. XỬ LÝ DATE RANGE
# ============================================================

if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        start_date = date_range[0]
        end_date = date_range[1]
    else:
        start_date = date_range[0]
        end_date = date_range[0]
else:
    start_date = date_range
    end_date = date_range

end_exclusive = end_date + timedelta(days=1)

start_iso = f"{start_date.isoformat()}T00:00:00Z"
end_iso = f"{end_exclusive.isoformat()}T00:00:00Z"


# ============================================================
# 12. ĐIỀU KIỆN SITE
# ============================================================

if selected_site == "Tất cả":
    site_condition = ""
else:
    safe_site = selected_site.replace("'", "''")
    site_condition = f"AND site_id = '{safe_site}'"


# ============================================================
# 13. HÀM TIỆN ÍCH
# ============================================================

def fmt_number(value, digits=2):
    """Định dạng số an toàn để hiển thị trên giao diện."""
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def build_page_header(eyebrow, title, description):
    """Hiển thị tiêu đề cho từng trang."""
    st.markdown(
        f"""
        <div class="page-content">
            <div class="page-eyebrow">{eyebrow}</div>
            <div class="page-title">{title}</div>
            <div class="page-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_base_layout(fig, height=400):
    """Thiết lập layout thống nhất cho Plotly."""
    fig.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=25,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )


def show_filter_summary():
    """Hiển thị tóm tắt bộ lọc hiện tại."""
    site_text = (
        "Tất cả các site"
        if selected_site == "Tất cả"
        else selected_site
    )

    st.caption(
        f"Phạm vi: {start_date.strftime('%d/%m/%Y')} → "
        f"{end_date.strftime('%d/%m/%Y')} · Site: {site_text}"
    )


# ============================================================
# 14. LOAD KPI
# ============================================================

@st.cache_data(ttl=60)
def get_kpi_data(start_iso_value, end_iso_value, site_condition_value):
    sql = f"""
        SELECT
            SUM(actual) AS total_actual,
            SUM(prediction) AS total_prediction,
            AVG(absolute_error) AS mae,
            SQRT(
                AVG(
                    POWER(error, 2)
                )
            ) AS rmse,
            COUNT(*) AS records
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso_value}'
            AND time < '{end_iso_value}'
            {site_condition_value}
    """

    return run_query(sql)


# ============================================================
# 15. LOAD MAIN DATASETS
#
# Các query được tách riêng để có thể dùng lại trên nhiều trang.
# ============================================================

@st.cache_data(ttl=60)
def get_trend_data(start_iso_value, end_iso_value, site_condition_value):
    sql = f"""
        SELECT
            DATE_TRUNC('hour', time) AS timestamp,
            SUM(actual) AS actual,
            SUM(prediction) AS prediction
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso_value}'
            AND time < '{end_iso_value}'
            {site_condition_value}
        GROUP BY 1
        ORDER BY 1
    """

    return run_query(sql)


@st.cache_data(ttl=60)
def get_hourly_data(start_iso_value, end_iso_value, site_condition_value):
    sql = f"""
        SELECT
            DATE_PART('hour', time) AS hour,
            AVG(actual) AS actual,
            AVG(prediction) AS prediction
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso_value}'
            AND time < '{end_iso_value}'
            {site_condition_value}
        GROUP BY 1
        ORDER BY 1
    """

    return run_query(sql)


@st.cache_data(ttl=60)
def get_site_data(start_iso_value, end_iso_value, site_condition_value):
    sql = f"""
        SELECT
            site_id,
            SUM(actual) AS actual,
            SUM(prediction) AS prediction,
            AVG(absolute_error) AS mae
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso_value}'
            AND time < '{end_iso_value}'
            {site_condition_value}
        GROUP BY site_id
        ORDER BY actual DESC
    """

    return run_query(sql)


@st.cache_data(ttl=60)
def get_daily_data(start_iso_value, end_iso_value, site_condition_value):
    sql = f"""
        SELECT
            DATE_TRUNC('day', time) AS date,
            SUM(actual) AS actual,
            SUM(prediction) AS prediction,
            AVG(absolute_error) AS mae
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso_value}'
            AND time < '{end_iso_value}'
            {site_condition_value}
        GROUP BY 1
        ORDER BY 1
    """

    return run_query(sql)


@st.cache_data(ttl=60)
def get_outlier_data(start_iso_value, end_iso_value, site_condition_value):
    sql = f"""
        SELECT
            is_outlier,
            COUNT(*) AS records,
            AVG(absolute_error) AS mae,
            SQRT(
                AVG(
                    POWER(error, 2)
                )
            ) AS rmse,
            MAX(absolute_error) AS max_error,
            SUM(
                POWER(error, 2)
            ) AS sse
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso_value}'
            AND time < '{end_iso_value}'
            {site_condition_value}
        GROUP BY is_outlier
        ORDER BY is_outlier
    """

    return run_query(sql)


@st.cache_data(ttl=60)
def get_top_errors(start_iso_value, end_iso_value, site_condition_value):
    sql = f"""
        SELECT
            time,
            site_id,
            actual,
            prediction,
            absolute_error,
            is_outlier
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso_value}'
            AND time < '{end_iso_value}'
            {site_condition_value}
        ORDER BY absolute_error DESC
        LIMIT 20
    """

    return run_query(sql)


# ============================================================
# 16. TRANG OVERVIEW
# ============================================================

def render_overview():
    build_page_header(
        "Overview",
        "Tổng quan hệ thống",
        "Tóm tắt dữ liệu, kết quả dự báo và các chỉ số chính của hệ thống.",
    )

    show_filter_summary()

    kpi_df = get_kpi_data(
        start_iso,
        end_iso,
        site_condition,
    )

    if kpi_df.empty:
        st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")
        return

    kpi = kpi_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Tiêu thụ thực tế",
            fmt_number(kpi["total_actual"], 0),
        )

    with col2:
        st.metric(
            "Tiêu thụ dự báo",
            fmt_number(kpi["total_prediction"], 0),
        )

    with col3:
        st.metric(
            "MAE",
            fmt_number(kpi["mae"], 2),
        )

    with col4:
        st.metric(
            "Số bản ghi",
            f"{int(kpi['records']):,}",
        )

    st.markdown(
        '<div class="section-title">Actual vs Predicted</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">So sánh mức tiêu thụ thực tế và giá trị dự báo theo thời gian.</div>',
        unsafe_allow_html=True,
    )

    trend_df = get_trend_data(
        start_iso,
        end_iso,
        site_condition,
    )

    if not trend_df.empty:
        trend_df["timestamp"] = pd.to_datetime(
            trend_df["timestamp"]
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trend_df["timestamp"],
                y=trend_df["actual"],
                mode="lines",
                name="Actual",
                line=dict(width=2.2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=trend_df["timestamp"],
                y=trend_df["prediction"],
                mode="lines",
                name="Predicted",
                line=dict(width=2),
            )
        )

        build_base_layout(fig, height=460)

        fig.update_layout(
            xaxis_title="Thời gian",
            yaxis_title="Điện năng tiêu thụ",
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )
    else:
        st.info("Chưa có dữ liệu biểu đồ.")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Dataset</div>
                <div class="info-card-text">
                    Building Data Genome Project 2 (BDG2),
                    dữ liệu điện năng kết hợp metadata và weather.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Big Data Processing</div>
                <div class="info-card-text">
                    PySpark được sử dụng cho preprocessing,
                    feature engineering và xử lý dữ liệu lớn.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Forecasting</div>
                <div class="info-card-text">
                    Ridge Regression được dùng cho bước dự báo
                    trên các đặc trưng time-series và weather.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Xu hướng theo ngày</div>',
        unsafe_allow_html=True,
    )

    daily_df = get_daily_data(
        start_iso,
        end_iso,
        site_condition,
    )

    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])

        fig_daily = go.Figure()

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["actual"],
                mode="lines",
                name="Actual",
                line=dict(width=2),
            )
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["prediction"],
                mode="lines",
                name="Predicted",
                line=dict(width=2),
            )
        )

        build_base_layout(fig_daily, height=390)

        fig_daily.update_layout(
            xaxis_title="Ngày",
            yaxis_title="Tổng tiêu thụ",
        )

        st.plotly_chart(
            fig_daily,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )


# ============================================================
# 17. TRANG ANALYSIS
# ============================================================

def render_analysis():
    build_page_header(
        "Analysis",
        "Phân tích tiêu thụ điện năng",
        "Khám phá mô hình tiêu thụ theo giờ, ngày và site.",
    )

    show_filter_summary()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            '<div class="section-title">Consumption by Hour</div>',
            unsafe_allow_html=True,
        )

        hourly_df = get_hourly_data(
            start_iso,
            end_iso,
            site_condition,
        )

        if not hourly_df.empty:
            hourly_df["hour"] = (
                hourly_df["hour"]
                .astype(int)
            )

            fig_hour = go.Figure()

            fig_hour.add_trace(
                go.Scatter(
                    x=hourly_df["hour"],
                    y=hourly_df["actual"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(width=2),
                )
            )

            fig_hour.add_trace(
                go.Scatter(
                    x=hourly_df["hour"],
                    y=hourly_df["prediction"],
                    mode="lines+markers",
                    name="Predicted",
                    line=dict(width=2),
                )
            )

            build_base_layout(fig_hour, height=370)

            fig_hour.update_layout(
                xaxis_title="Giờ trong ngày",
                yaxis_title="Mức tiêu thụ",
                xaxis=dict(dtick=2),
            )

            st.plotly_chart(
                fig_hour,
                width="stretch",
                config={"displaylogo": False},
            )

    with col_right:
        st.markdown(
            '<div class="section-title">Consumption by Site</div>',
            unsafe_allow_html=True,
        )

        site_df = get_site_data(
            start_iso,
            end_iso,
            site_condition,
        )

        if not site_df.empty:
            fig_site = go.Figure()

            fig_site.add_trace(
                go.Bar(
                    x=site_df["site_id"],
                    y=site_df["actual"],
                    name="Actual",
                )
            )

            fig_site.add_trace(
                go.Bar(
                    x=site_df["site_id"],
                    y=site_df["prediction"],
                    name="Predicted",
                )
            )

            build_base_layout(fig_site, height=370)

            fig_site.update_layout(
                barmode="group",
                xaxis_title="Site",
                yaxis_title="Tổng tiêu thụ",
            )

            st.plotly_chart(
                fig_site,
                width="stretch",
                config={"displaylogo": False},
            )

    st.markdown(
        '<div class="section-title">Consumption by Day</div>',
        unsafe_allow_html=True,
    )

    daily_df = get_daily_data(
        start_iso,
        end_iso,
        site_condition,
    )

    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(
            daily_df["date"]
        )

        fig_daily = go.Figure()

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["actual"],
                mode="lines",
                name="Actual",
                line=dict(width=2),
            )
        )

        fig_daily.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df["prediction"],
                mode="lines",
                name="Predicted",
                line=dict(width=2),
            )
        )

        build_base_layout(fig_daily, height=430)

        fig_daily.update_layout(
            xaxis_title="Ngày",
            yaxis_title="Tổng tiêu thụ",
        )

        st.plotly_chart(
            fig_daily,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    with st.expander("Xem bảng dữ liệu tổng hợp"):
        if not daily_df.empty:
            table_df = daily_df.copy()

            table_df["date"] = (
                table_df["date"]
                .dt.strftime("%Y-%m-%d")
            )

            table_df = table_df.rename(
                columns={
                    "date": "Ngày",
                    "actual": "Thực tế",
                    "prediction": "Dự báo",
                    "mae": "MAE",
                }
            )

            st.dataframe(
                table_df,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# 18. TRANG FORECAST
# ============================================================

def render_forecast():
    build_page_header(
        "Forecast",
        "Dự báo tiêu thụ điện năng",
        "Kết quả mô hình Ridge Regression trên tập kiểm tra năm 2017.",
    )

    show_filter_summary()

    model_col1, model_col2, model_col3, model_col4 = st.columns(4)

    with model_col1:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">Mô hình</div>
                <div class="model-info-value" style="font-size:22px;">
                    Ridge Regression
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with model_col2:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">RMSE</div>
                <div class="model-info-value">1,295.99</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with model_col3:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">MAE</div>
                <div class="model-info-value">383.17</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with model_col4:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">R²</div>
                <div class="model-info-value">0.98505</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.caption(
        "Ridge Regression · regParam = 10 · "
        "Đánh giá trên tập kiểm tra năm 2017"
    )

    st.markdown(
        '<div class="section-title">Actual vs Predicted</div>',
        unsafe_allow_html=True,
    )

    trend_df = get_trend_data(
        start_iso,
        end_iso,
        site_condition,
    )

    if not trend_df.empty:
        trend_df["timestamp"] = pd.to_datetime(
            trend_df["timestamp"]
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trend_df["timestamp"],
                y=trend_df["actual"],
                mode="lines",
                name="Actual",
                line=dict(width=2.2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=trend_df["timestamp"],
                y=trend_df["prediction"],
                mode="lines",
                name="Predicted",
                line=dict(width=2),
            )
        )

        build_base_layout(fig, height=460)

        fig.update_layout(
            xaxis_title="Thời gian",
            yaxis_title="Điện năng tiêu thụ",
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

    st.markdown(
        '<div class="section-title">Kết quả các mô hình</div>',
        unsafe_allow_html=True,
    )

    model_df = pd.DataFrame(
        {
            "Mô hình": [
                "Naive baseline",
                "Linear Regression",
                "Random Forest Regression",
                "Ridge Regression",
            ],
            "RMSE": [
                2171.919018,
                1296.931754,
                1357.076116,
                1295.986751,
            ],
            "MAE": [
                768.596259,
                377.429933,
                537.137581,
                383.172457,
            ],
            "R²": [
                0.958002,
                0.985025,
                0.983604,
                0.985047,
            ],
        }
    )

    st.dataframe(
        model_df.style.format(
            {
                "RMSE": "{:,.2f}",
                "MAE": "{:,.2f}",
                "R²": "{:.5f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Các mô hình được dùng để so sánh trong quá trình thực nghiệm."
    )

    st.markdown(
        '<div class="section-title">Sai số dự báo theo thời gian</div>',
        unsafe_allow_html=True,
    )

    error_sql = f"""
        SELECT
            DATE_TRUNC('hour', time) AS timestamp,
            SUM(absolute_error) AS absolute_error
        FROM "{MEASUREMENT}"
        WHERE
            time >= '{start_iso}'
            AND time < '{end_iso}'
            {site_condition}
        GROUP BY 1
        ORDER BY 1
    """

    error_df = run_query(error_sql)

    if not error_df.empty:
        error_df["timestamp"] = pd.to_datetime(
            error_df["timestamp"]
        )

        fig_error = go.Figure()

        fig_error.add_trace(
            go.Scatter(
                x=error_df["timestamp"],
                y=error_df["absolute_error"],
                mode="lines",
                name="Absolute Error",
                line=dict(width=1.8),
            )
        )

        build_base_layout(fig_error, height=390)

        fig_error.update_layout(
            xaxis_title="Thời gian",
            yaxis_title="Tổng sai số tuyệt đối",
        )

        st.plotly_chart(
            fig_error,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )


# ============================================================
# 19. TRANG ANOMALY
# ============================================================

def render_anomaly():
    build_page_header(
        "Anomaly Analysis",
        "Phân tích Outlier và sai số",
        "Theo dõi ảnh hưởng của các điểm bất thường đến chất lượng dự báo.",
    )

    show_filter_summary()

    outlier_df = get_outlier_data(
        start_iso,
        end_iso,
        site_condition,
    )

    if not outlier_df.empty:
        outlier_df["is_outlier"] = (
            outlier_df["is_outlier"]
            .astype(str)
            .str.strip()
        )

        normal_row = outlier_df[
            outlier_df["is_outlier"] == "0"
        ]

        outlier_row = outlier_df[
            outlier_df["is_outlier"] == "1"
        ]

        normal_mae = 0.0
        outlier_mae = 0.0
        outlier_sse_share = 0.0
        outlier_count = 0
        total_count = 0

        if not normal_row.empty:
            normal_mae = float(
                normal_row.iloc[0]["mae"]
            )

            total_count += int(
                normal_row.iloc[0]["records"]
            )

        if not outlier_row.empty:
            outlier_mae = float(
                outlier_row.iloc[0]["mae"]
            )

            outlier_count = int(
                outlier_row.iloc[0]["records"]
            )

            total_count += outlier_count

        total_sse = float(
            pd.to_numeric(
                outlier_df["sse"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        if total_sse > 0 and not outlier_row.empty:
            outlier_sse = float(
                pd.to_numeric(
                    outlier_row.iloc[0]["sse"],
                    errors="coerce",
                )
            )

            outlier_sse_share = (
                outlier_sse / total_sse * 100
            )

        out_col1, out_col2, out_col3, out_col4 = st.columns(4)

        with out_col1:
            st.metric(
                "MAE - Normal",
                f"{normal_mae:,.2f}",
            )

        with out_col2:
            st.metric(
                "MAE - Outlier",
                f"{outlier_mae:,.2f}",
            )

        with out_col3:
            st.metric(
                "Outlier SSE",
                f"{outlier_sse_share:.2f}%",
            )

        with out_col4:
            outlier_rate = (
                outlier_count / total_count * 100
                if total_count > 0
                else 0
            )

            st.metric(
                "Tỷ lệ Outlier",
                f"{outlier_rate:.2f}%",
            )

        st.caption(
            "Outlier không bị xóa khỏi dữ liệu; "
            "chỉ được dùng để phân tích sai số."
        )
    else:
        st.info("Không có dữ liệu outlier trong khoảng thời gian đã chọn.")

    st.markdown(
        '<div class="section-title">Top 20 Prediction Errors</div>',
        unsafe_allow_html=True,
    )

    top_error_df = get_top_errors(
        start_iso,
        end_iso,
        site_condition,
    )

    if not top_error_df.empty:
        display_df = top_error_df.copy()

        display_df["time"] = (
            pd.to_datetime(display_df["time"])
            .dt.strftime("%Y-%m-%d %H:%M")
        )

        display_df["is_outlier"] = (
            display_df["is_outlier"]
            .astype(str)
            .str.strip()
        )

        display_df = display_df.rename(
            columns={
                "time": "Thời gian",
                "site_id": "Site",
                "actual": "Thực tế",
                "prediction": "Dự báo",
                "absolute_error": "Sai số tuyệt đối",
                "is_outlier": "Outlier",
            }
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "Không có dữ liệu sai số trong khoảng thời gian đã chọn."
        )


# ============================================================
# 20. TRANG ABOUT
# ============================================================

def render_about():
    build_page_header(
        "About",
        "Giới thiệu đồ án",
        "Electricity Consumption Analytics — đồ án môn Nhập môn Big Data.",
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Mục tiêu</div>
            <div class="info-card-text">
                Phân tích dữ liệu tiêu thụ điện năng theo thời gian,
                xử lý dữ liệu lớn bằng Apache Spark và xây dựng mô hình
                dự báo nhu cầu điện năng.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Công nghệ sử dụng</div>',
        unsafe_allow_html=True,
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Data Processing</div>
                <div class="info-card-text">
                    Python · Apache Spark · PySpark · Parquet
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Machine Learning</div>
                <div class="info-card-text">
                    Spark ML · Linear Regression · Random Forest · Ridge Regression
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Data & Visualization</div>
                <div class="info-card-text">
                    InfluxDB · Streamlit · Plotly
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Quy trình xử lý</div>',
        unsafe_allow_html=True,
    )

    pipeline_items = [
        ("01", "Raw Data"),
        ("02", "Preprocessing"),
        ("03", "Feature Engineering"),
        ("04", "Train / Validation / Test"),
        ("05", "Spark ML"),
        ("06", "Prediction"),
        ("07", "InfluxDB"),
        ("08", "Dashboard"),
    ]

    pipeline_cols = st.columns(4)

    for index, (number, name) in enumerate(pipeline_items):
        with pipeline_cols[index % 4]:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<div class="pipeline-number">{number}</div>'
                f'<div class="pipeline-name">{name}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Kiến trúc hệ thống</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">Luồng dữ liệu từ dataset đến dashboard phục vụ phân tích và dự báo.</div>',
        unsafe_allow_html=True,
    )

    architecture_items = [
        ("01", "BDG2 Raw Data"),
        ("02", "PySpark Processing"),
        ("03", "Feature Engineering"),
        ("04", "Spark ML"),
        ("05", "Prediction"),
        ("06", "InfluxDB"),
        ("07", "Streamlit"),
        ("08", "Interactive Dashboard"),
    ]

    arch_cols = st.columns(4)

    for index, (number, name) in enumerate(architecture_items):
        with arch_cols[index % 4]:
            st.markdown(
                f"""
                <div class="architecture-step">
                    <div class="architecture-number">{number}</div>
                    <div class="architecture-name">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Thông tin dữ liệu</div>',
        unsafe_allow_html=True,
    )

    data1, data2, data3, data4 = st.columns(4)

    with data1:
        st.metric("Buildings sau làm sạch", "1,514")

    with data2:
        st.metric("Sites", "18")

    with data3:
        st.metric("Electricity records", "25.95M")

    with data4:
        st.metric("Test predictions", "153,016")

    st.markdown(
        '<div class="section-title">Mô hình cuối</div>',
        unsafe_allow_html=True,
    )

    model1, model2, model3, model4 = st.columns(4)

    with model1:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">Model</div>
                <div class="model-info-value" style="font-size:22px;">
                    Ridge Regression
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with model2:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">regParam</div>
                <div class="model-info-value">10</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with model3:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">RMSE</div>
                <div class="model-info-value">1,295.99</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with model4:
        st.markdown(
            '''
            <div class="model-info-card">
                <div class="model-info-label">R²</div>
                <div class="model-info-value">0.98505</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.caption(
        "Các kết quả được sử dụng trong dashboard là kết quả thực nghiệm "
        "đã hoàn thành của project."
    )


# ============================================================
# 21. ĐIỀU HƯỚNG TRANG
# ============================================================

if page == "Overview":
    render_overview()

elif page == "Analysis":
    render_analysis()

elif page == "Forecast":
    render_forecast()

elif page == "Anomaly":
    render_anomaly()

elif page == "About":
    render_about()


# ============================================================
# 22. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Electricity Consumption Analytics · Apache Spark · Spark ML · InfluxDB · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
