from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils.chart_utils import (
    create_booked_vs_walkin_chart,
    create_correlation_regression_chart,
    create_daywise_analysis_chart,
    create_department_analysis_chart,
    create_hourly_analysis_chart,
    create_load_ratio_distribution_chart,
    create_monte_carlo_histogram,
    create_peak_probability_chart,
    create_scenario_comparison_chart,
    create_simulation_queue_chart,
    create_simulation_waiting_time_chart,
    create_waiting_time_distribution_chart,
)
from utils.data_loader import (
    ensure_datasets_available,
    get_basic_dataset_info,
    get_dataset_preview,
    load_all_datasets,
)
from utils.predictor import (
    get_prediction_reference_ranges,
    predict_overload_risk,
    predict_overload_risk_by_department,
    predict_waiting_time_by_department,
    predict_waiting_time_from_patient_count,
)
from utils.probability_utils import (
    calculate_department_peak_probability,
    calculate_high_waiting_time_probability,
    calculate_overcrowding_probability,
    calculate_peak_day_probability,
    calculate_peak_hour_probability_by_hour,
    calculate_waiting_time_probability_by_department,
    calculate_walk_in_vs_booked_probability,
    calculate_walk_in_vs_booked_probability_by_department,
    get_department_poisson_probability_summary,
    get_filtered_poisson_probability_summary,
)
from utils.simulation_utils import (
    compare_scenarios,
    get_available_days,
    get_simulation_summary,
    monte_carlo_simulation,
    simulate_queue_for_day,
)
from utils.stats_utils import (
    calculate_correlation,
    calculate_descriptive_statistics,
    get_daywise_patient_analysis,
    get_department_analysis,
    get_hourly_patient_analysis,
    get_numeric_columns,
    get_peak_time_summary,
    get_summary_statistics_table,
    get_waiting_time_summary,
    perform_linear_regression,
)


st.set_page_config(
    page_title="OPD Patient Arrival Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATASET_LABELS = {
    "opd_patient_data": "Main OPD Patient Arrivals",
    "department_wise_opd_data": "Department-wise OPD Data",
    "doctor_schedule_data": "Doctor Schedule Data",
    "appointment_data": "Appointment Data",
    "patient_category_data": "Patient Category Data",
}


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text-main: #0f172a;
            --text-soft: #475569;
            --panel: rgba(255,255,255,0.94);
            --panel-border: rgba(148,163,184,0.18);
        }

        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 50%, #f8fbff 100%);
            color: var(--text-main);
        }

        section.main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1.2rem;
        }

        section.main,
        section.main p,
        section.main li,
        section.main label,
        section.main .stMarkdown,
        section.main h1,
        section.main h2,
        section.main h3,
        section.main h4,
        section.main h5,
        section.main h6,
        section.main div[data-testid="stMarkdownContainer"] {
            color: var(--text-main) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        [data-testid="stSidebar"] * {
            color: #e5eefc !important;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #38bdf8 100%);
            border-radius: 24px;
            padding: 28px 30px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }

        .hero-card,
        .hero-card * {
            color: white !important;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.45rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: rgba(255,255,255,0.88) !important;
            margin-bottom: 0.9rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 12px;
        }

        .pill {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.84rem;
        }

        .section-block {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            padding: 18px 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a !important;
            margin-bottom: 0.25rem;
        }

        .section-subtitle {
            color: #475569 !important;
            font-size: 0.95rem;
            margin-bottom: 0;
        }

        .stat-card {
            background: white;
            border-radius: 22px;
            padding: 18px;
            border: 1px solid var(--panel-border);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            min-height: 128px;
        }

        .stat-label {
            color: #64748b !important;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .stat-value {
            color: #0f172a !important;
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 8px;
        }

        .stat-caption {
            color: #475569 !important;
            font-size: 0.85rem;
        }

        .mini-card,
        .info-card {
            background: white;
            border-radius: 20px;
            padding: 16px 18px;
            border: 1px solid var(--panel-border);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .info-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0f172a !important;
            margin-bottom: 10px;
        }

        .info-list {
            margin: 0;
            padding-left: 1rem;
            color: #334155 !important;
        }

        .info-list li {
            margin-bottom: 7px;
        }

        .chip {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .chip-blue { background: #dbeafe; color: #1d4ed8 !important; }
        .chip-purple { background: #ede9fe; color: #7c3aed !important; }
        .chip-green { background: #dcfce7; color: #15803d !important; }
        .chip-amber { background: #fef3c7; color: #b45309 !important; }
        .chip-rose { background: #ffe4e6; color: #be123c !important; }

        .risk-pill {
            display: inline-block;
            padding: 7px 13px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 800;
        }

        .risk-low { background: #dcfce7; color: #166534 !important; }
        .risk-moderate { background: #fef3c7; color: #92400e !important; }
        .risk-high { background: #fed7aa; color: #c2410c !important; }
        .risk-critical { background: #ffe4e6; color: #be123c !important; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff !important;
            border-radius: 14px 14px 0 0 !important;
            border: 1px solid rgba(148,163,184,0.22) !important;
            padding: 10px 16px !important;
            color: #334155 !important;
        }

        .stTabs [data-baseweb="tab"] * {
            color: #334155 !important;
        }

        .stTabs [aria-selected="true"] {
            background: #dbeafe !important;
            color: #1d4ed8 !important;
            border-bottom: 3px solid #2563eb !important;
        }

        .stTabs [aria-selected="true"] * {
            color: #1d4ed8 !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--panel-border);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
            background: white;
        }

        div[data-testid="stSelectbox"],
        div[data-testid="stNumberInput"],
        div[data-testid="stSlider"] {
            background: transparent;
            color: #0f172a !important;
        }

        .footer-note {
            text-align: center;
            color: #64748b !important;
            font-size: 0.88rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def format_probability(probability_value: float) -> str:
    return f"{float(probability_value) * 100:.2f}%"


def show_matplotlib_figure(fig) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-block">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, items: list[str]) -> None:
    list_items = "".join([f"<li>{item}</li>" for item in items])
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">{title}</div>
            <ul class="info-list">
                {list_items}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_pill(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized == "low":
        css_class = "risk-low"
    elif normalized == "moderate":
        css_class = "risk-moderate"
    elif normalized == "high":
        css_class = "risk-high"
    else:
        css_class = "risk-critical"

    return f'<span class="risk-pill {css_class}">{label}</span>'


def top_peak_hour(peak_hour_df: pd.DataFrame) -> str:
    if peak_hour_df.empty:
        return "N/A"
    row = peak_hour_df.sort_values(
        ["peak_probability", "average_waiting_time", "average_patients"],
        ascending=[False, False, False],
    ).iloc[0]
    return str(row["hour_slot"])


def top_peak_day(peak_day_df: pd.DataFrame) -> str:
    if peak_day_df.empty:
        return "N/A"
    row = peak_day_df.sort_values(
        ["peak_probability", "average_waiting_time", "average_patients"],
        ascending=[False, False, False],
    ).iloc[0]
    return str(row["day"])


@st.cache_data(show_spinner=False)
def load_project_data() -> Dict[str, pd.DataFrame]:
    ensure_datasets_available()
    return load_all_datasets()


def render_home_section(
    opd_df: pd.DataFrame,
    department_df: pd.DataFrame,
    appointment_df: pd.DataFrame,
) -> None:
    peak_summary = get_peak_time_summary(opd_df)
    wait_summary = get_waiting_time_summary(opd_df)

    peak_hour_df = calculate_peak_hour_probability_by_hour(opd_df)
    peak_day_df = calculate_peak_day_probability(opd_df)
    department_peak_df = calculate_department_peak_probability(department_df)
    booking_summary = calculate_walk_in_vs_booked_probability(appointment_df)

    total_patients = int(pd.to_numeric(opd_df["patients_arrived"], errors="coerce").fillna(0).sum())
    total_records = int(len(opd_df))
    total_departments = int(department_df["department"].nunique())
    avg_waiting_time = float(pd.to_numeric(opd_df["avg_waiting_time"], errors="coerce").fillna(0).mean())

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">🏥 OPD Patient Arrival Intelligence Dashboard</div>
            <div class="hero-subtitle">
                Statistical analysis, probability modelling, queue simulation, and prediction for smarter appointment planning.
            </div>
            <div class="pill-row">
                <span class="pill">📈 Descriptive Statistics</span>
                <span class="pill">🎯 Probability Analysis</span>
                <span class="pill">🧠 Prediction</span>
                <span class="pill">🧪 Simulation</span>
                <span class="pill">📝 Report-Ready Insights</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        render_stat_card("Total OPD Patients", f"{total_patients:,}", "Synthetic but realistic hospital flow")
    with cols[1]:
        render_stat_card("Hourly Records", f"{total_records:,}", "System-wide hourly observations")
    with cols[2]:
        render_stat_card("Departments Covered", f"{total_departments}", "Multi-department OPD view")
    with cols[3]:
        render_stat_card("Average Waiting Time", f"{avg_waiting_time:.2f} min", "Across all OPD observations")

    render_section_header(
        "Operational Snapshot",
        "A fast pulse-check of where the OPD system gets crowded, delayed, or appointment-heavy.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        render_info_card(
            "Peak Pattern Highlights",
            [
                f"Busiest hour from volume analysis: {peak_summary['busiest_hour']}",
                f"Most likely peak hour by probability: {top_peak_hour(peak_hour_df)}",
                f"Busiest day from volume analysis: {peak_summary['busiest_day']}",
                f"Most likely peak day by probability: {top_peak_day(peak_day_df)}",
            ],
        )
    with c2:
        top_dept = department_peak_df.iloc[0]["department"] if not department_peak_df.empty else "N/A"
        render_info_card(
            "Risk Highlights",
            [
                f"Overall peak occurrence rate: {format_probability(peak_summary['overall_peak_rate'])}",
                f"High waiting time rate: {format_probability(wait_summary['high_waiting_time_rate'])}",
                f"Department with strongest peak tendency: {top_dept}",
                f"Maximum waiting time observed: {wait_summary['maximum_waiting_time']:.2f} min",
            ],
        )
    with c3:
        render_info_card(
            "Appointment Behavior",
            [
                f"Booked probability: {format_probability(booking_summary['booked_probability'])}",
                f"Walk-in probability: {format_probability(booking_summary['walk_in_probability'])}",
                "Walk-ins can disrupt planned capacity even when booked slots look balanced",
                "Department-level variation matters for queue build-up",
            ],
        )

    render_section_header(
        "What this project helps explain",
        "This is the academic story your dashboard tells without needing ten pages of dry narration.",
    )

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown(
            """
            <div class="mini-card">
                <span class="chip chip-blue">Hour-wise Analysis</span>
                <span class="chip chip-purple">Day-wise Trends</span>
                <span class="chip chip-green">Department Pressure</span>
                <span class="chip chip-amber">Queue Simulation</span>
                <span class="chip chip-rose">Poisson Arrival Estimation</span>
                <p style="margin-top:12px; color:#334155;">
                The dashboard studies how patient arrival patterns shift across time, how those shifts affect waiting time and load,
                and how simulation can support better staffing and appointment planning decisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        min_date = opd_df["date"].min()
        max_date = opd_df["date"].max()
        date_text = (
            f"{min_date.date()} to {max_date.date()}"
            if pd.notna(min_date) and pd.notna(max_date)
            else "Available in dataset"
        )
        render_info_card(
            "Dataset Window",
            [
                f"Date coverage: {date_text}",
                "Synthetic datasets are auto-generated locally",
                "Designed for stable PyCharm + Streamlit execution",
                "No fragile external APIs or cloud dependency",
            ],
        )
    st.markdown(
        """
        <div class="section-block" style="margin-top: 1.5rem;">
            <div class="section-title">Experiential Learning Project</div>
            <div class="section-subtitle" style="margin-bottom: 12px;">
                Statistics and Probability
            </div>
            <div style="color:#334155; font-size:1rem; line-height:1.9;">
                <strong>Group Members:</strong><br>
                Prajwal C Pradhan (Team lead)<br>
                Namgay D Wangchuk<br>
                Malavika Vinod<br>
                Monish Kandanuru<br>
                P. Navya Shree<br>
                Mumukka Sanjana Reddy<br>
                Pakalapati Monika
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_dataset_overview_section(all_data: Dict[str, pd.DataFrame]) -> None:
    render_section_header(
        "Dataset Overview",
        "Inspect each dataset, verify its columns, and confirm the system has loaded everything cleanly.",
    )

    dataset_key = st.selectbox(
        "Select dataset",
        options=list(DATASET_LABELS.keys()),
        format_func=lambda x: DATASET_LABELS[x],
    )

    selected_df = all_data[dataset_key]
    dataset_info = get_basic_dataset_info(selected_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_stat_card("Dataset", DATASET_LABELS[dataset_key], "Currently selected")
    with c2:
        render_stat_card("Rows", str(dataset_info["rows"]), "Volume of observations")
    with c3:
        render_stat_card("Columns", str(dataset_info["columns"]), "Fields available for analysis")

    st.markdown("### Preview")
    st.dataframe(get_dataset_preview(selected_df, num_rows=10), use_container_width=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Column Names")
        st.write(dataset_info["column_names"])

    with col2:
        st.markdown("### Missing Values")
        missing_df = pd.DataFrame(
            {
                "column": list(dataset_info["missing_values"].keys()),
                "missing_values": list(dataset_info["missing_values"].values()),
            }
        )
        st.dataframe(missing_df, use_container_width=True)

    with st.expander("Show full dataset table"):
        st.dataframe(selected_df, use_container_width=True, height=400)


def render_statistical_analysis_section(
    opd_df: pd.DataFrame,
    department_df: pd.DataFrame,
) -> None:
    render_section_header(
        "Statistical Analysis",
        "Crunch the numerical skeleton of the OPD system: central tendency, spread, operational patterns, and regression relationships.",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Descriptive Statistics",
            "Operational Patterns",
            "Correlation and Regression",
        ]
    )

    with tab1:
        st.markdown("### Descriptive Statistics")

        stats_dataset_key = st.selectbox(
            "Select dataset for statistics",
            options=["opd_patient_data", "department_wise_opd_data"],
            format_func=lambda x: DATASET_LABELS[x],
            key="stats_dataset_key",
        )

        stats_df = opd_df if stats_dataset_key == "opd_patient_data" else department_df
        numeric_columns = get_numeric_columns(stats_df)

        selected_columns = st.multiselect(
            "Select numeric columns",
            options=numeric_columns,
            default=[col for col in ["patients_arrived", "avg_waiting_time", "load_ratio"] if col in numeric_columns],
        )

        if selected_columns:
            summary_table = get_summary_statistics_table(stats_df, selected_columns)
            st.dataframe(summary_table, use_container_width=True)

            single_column = st.selectbox(
                "View detailed stats for one column",
                options=selected_columns,
                key="single_stats_column",
            )
            detailed_stats = calculate_descriptive_statistics(stats_df, single_column)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_stat_card("Mean", f"{detailed_stats['mean']}", f"{single_column}")
            with c2:
                render_stat_card("Median", f"{detailed_stats['median']}", f"{single_column}")
            with c3:
                render_stat_card("Std. Deviation", f"{detailed_stats['standard_deviation']}", f"{single_column}")
            with c4:
                render_stat_card("Range", f"{detailed_stats['range']}", f"{single_column}")

            detail_df = pd.DataFrame(
                {
                    "Metric": list(detailed_stats.keys()),
                    "Value": list(detailed_stats.values()),
                }
            )
            st.dataframe(detail_df, use_container_width=True)

            if single_column == "avg_waiting_time":
                fig = create_waiting_time_distribution_chart(stats_df)
                show_matplotlib_figure(fig)
            elif single_column == "load_ratio":
                fig = create_load_ratio_distribution_chart(stats_df)
                show_matplotlib_figure(fig)

    with tab2:
        st.markdown("### Hour-wise, Day-wise, and Department-wise Patterns")

        hourly_df = get_hourly_patient_analysis(opd_df)
        daywise_df = get_daywise_patient_analysis(opd_df)
        department_analysis_df = get_department_analysis(department_df)

        st.markdown("#### Hour-wise Patient Analysis")
        st.dataframe(hourly_df, use_container_width=True)
        fig = create_hourly_analysis_chart(hourly_df)
        show_matplotlib_figure(fig)

        st.markdown("#### Day-wise Patient Analysis")
        st.dataframe(daywise_df, use_container_width=True)
        fig = create_daywise_analysis_chart(daywise_df)
        show_matplotlib_figure(fig)

        st.markdown("#### Department-wise Analysis")
        st.dataframe(department_analysis_df, use_container_width=True)
        fig = create_department_analysis_chart(department_analysis_df)
        show_matplotlib_figure(fig)

    with tab3:
        st.markdown("### Correlation and Linear Regression")

        regression_dataset_key = st.selectbox(
            "Select dataset for regression",
            options=["opd_patient_data", "department_wise_opd_data"],
            format_func=lambda x: DATASET_LABELS[x],
            key="regression_dataset_key",
        )

        regression_df = opd_df if regression_dataset_key == "opd_patient_data" else department_df
        numeric_columns = get_numeric_columns(regression_df)

        default_x = "patients_arrived" if "patients_arrived" in numeric_columns else numeric_columns[0]
        default_y = "avg_waiting_time" if "avg_waiting_time" in numeric_columns else numeric_columns[-1]

        x_column = st.selectbox(
            "X-axis column",
            options=numeric_columns,
            index=numeric_columns.index(default_x),
        )
        y_column = st.selectbox(
            "Y-axis column",
            options=numeric_columns,
            index=numeric_columns.index(default_y),
        )

        if x_column == y_column:
            st.warning("Choose two different columns for regression.")
        else:
            correlation_value = calculate_correlation(regression_df, x_column, y_column)
            regression_result = perform_linear_regression(regression_df, x_column, y_column)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_stat_card("Correlation", f"{correlation_value:.4f}", "Linear relationship strength")
            with c2:
                render_stat_card("Slope", f"{regression_result['slope']:.4f}", "Rate of change")
            with c3:
                render_stat_card("Intercept", f"{regression_result['intercept']:.4f}", "Regression intercept")
            with c4:
                render_stat_card("R²", f"{regression_result['r_squared']:.4f}", "Model fit")

            if x_column == "patients_arrived" and y_column == "avg_waiting_time":
                fig = create_correlation_regression_chart(regression_df)
                show_matplotlib_figure(fig)
            else:
                st.info(
                    "The visual regression chart is optimized for Patients Arrived vs Average Waiting Time. "
                    "Your selected variables are still analyzed numerically above."
                )


def render_probability_and_prediction_section(
    opd_df: pd.DataFrame,
    department_df: pd.DataFrame,
    appointment_df: pd.DataFrame,
) -> None:
    render_section_header(
        "Probability and Prediction",
        "Estimate overload, predict waiting time, and understand how likely the OPD system is to drift into stressful territory.",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Prediction",
            "Probability Analysis",
            "Poisson and Appointment Insights",
        ]
    )

    with tab1:
        st.markdown("### Waiting Time and Overload Prediction")

        reference_ranges = get_prediction_reference_ranges(opd_df)
        available_departments = sorted(department_df["department"].dropna().unique().tolist())

        scope = st.selectbox(
            "Prediction scope",
            options=["Overall OPD", "Specific Department"],
        )

        patient_count = st.slider(
            "Expected patient count",
            min_value=max(1, int(reference_ranges["min_patients"])),
            max_value=max(int(reference_ranges["max_patients"]) + 20, int(reference_ranges["min_patients"]) + 10),
            value=int(round(reference_ranges["average_patients"])),
        )

        doctors_available = st.number_input(
            "Doctors available",
            min_value=1,
            max_value=50,
            value=max(1, int(round(reference_ranges["average_doctors_available"]))),
            step=1,
        )

        if scope == "Overall OPD":
            waiting_prediction = predict_waiting_time_from_patient_count(opd_df, patient_count)
            overload_prediction = predict_overload_risk(opd_df, patient_count, doctors_available=doctors_available)

            c1, c2, c3 = st.columns(3)
            with c1:
                render_stat_card("Predicted Waiting Time", f"{waiting_prediction['predicted_waiting_time']:.2f} min", "From regression model")
            with c2:
                render_stat_card("Predicted Load Ratio", f"{overload_prediction['predicted_load_ratio']:.2f}", "Demand vs capacity")
            with c3:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">Overall Risk</div>
                        <div style="margin-top:16px;">{render_risk_pill(overload_prediction['overall_risk_label'])}</div>
                        <div class="stat-caption" style="margin-top:14px;">Combined load + waiting pressure</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Input Patient Count",
                            "Doctors Available",
                            "Estimated Capacity",
                            "Predicted Waiting Time",
                            "Predicted Load Ratio",
                            "Historical Overload Probability",
                            "Load Risk Label",
                            "Waiting Risk Label",
                            "Overall Risk Label",
                            "Regression Correlation",
                            "Regression R²",
                        ],
                        "Value": [
                            overload_prediction["input_patient_count"],
                            overload_prediction["doctors_available"],
                            overload_prediction["estimated_capacity"],
                            overload_prediction["predicted_waiting_time"],
                            overload_prediction["predicted_load_ratio"],
                            overload_prediction["historical_overload_probability"],
                            overload_prediction["load_risk_label"],
                            overload_prediction["waiting_risk_label"],
                            overload_prediction["overall_risk_label"],
                            waiting_prediction["correlation"],
                            waiting_prediction["r_squared"],
                        ],
                    }
                ),
                use_container_width=True,
            )

        else:
            selected_department = st.selectbox(
                "Select department",
                options=available_departments,
            )

            waiting_prediction = predict_waiting_time_by_department(
                department_df,
                selected_department,
                patient_count,
            )
            overload_prediction = predict_overload_risk_by_department(
                department_df,
                selected_department,
                patient_count,
                doctors_available=doctors_available,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                render_stat_card("Predicted Waiting Time", f"{waiting_prediction['predicted_waiting_time']:.2f} min", selected_department)
            with c2:
                render_stat_card("Predicted Load Ratio", f"{overload_prediction['predicted_load_ratio']:.2f}", selected_department)
            with c3:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">Overall Risk</div>
                        <div style="margin-top:16px;">{render_risk_pill(overload_prediction['overall_risk_label'])}</div>
                        <div class="stat-caption" style="margin-top:14px;">Department-specific stress label</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Department",
                            "Input Patient Count",
                            "Doctors Available",
                            "Estimated Capacity",
                            "Predicted Waiting Time",
                            "Predicted Load Ratio",
                            "Historical Overload Probability",
                            "Load Risk Label",
                            "Waiting Risk Label",
                            "Overall Risk Label",
                            "Regression Correlation",
                            "Regression R²",
                        ],
                        "Value": [
                            selected_department,
                            overload_prediction["input_patient_count"],
                            overload_prediction["doctors_available"],
                            overload_prediction["estimated_capacity"],
                            overload_prediction["predicted_waiting_time"],
                            overload_prediction["predicted_load_ratio"],
                            overload_prediction["historical_overload_probability"],
                            overload_prediction["load_risk_label"],
                            overload_prediction["waiting_risk_label"],
                            overload_prediction["overall_risk_label"],
                            waiting_prediction["correlation"],
                            waiting_prediction["r_squared"],
                        ],
                    }
                ),
                use_container_width=True,
            )

    with tab2:
        st.markdown("### Historical Probability Analysis")

        overcrowding_probability = calculate_overcrowding_probability(opd_df)
        high_wait_probability = calculate_high_waiting_time_probability(opd_df)
        peak_hour_df = calculate_peak_hour_probability_by_hour(opd_df)
        peak_day_df = calculate_peak_day_probability(opd_df)
        department_peak_df = calculate_department_peak_probability(department_df)
        waiting_probability_dept_df = calculate_waiting_time_probability_by_department(department_df)

        c1, c2 = st.columns(2)
        with c1:
            render_stat_card("Overcrowding Probability", format_probability(overcrowding_probability), "Chance that load ratio crosses threshold")
        with c2:
            render_stat_card("High Waiting Time Probability", format_probability(high_wait_probability), "Chance waiting time is 30 min or more")

        st.markdown("#### Peak-Hour Probability by Hour Slot")
        st.dataframe(peak_hour_df, use_container_width=True)
        fig = create_peak_probability_chart(
            peak_hour_df,
            label_column="hour_slot",
            title="Peak Probability by Hour Slot",
            x_label="Hour Slot",
            y_label="Peak Probability",
            rotate_x=True,
        )
        show_matplotlib_figure(fig)

        st.markdown("#### Peak Probability by Day")
        st.dataframe(peak_day_df, use_container_width=True)
        fig = create_peak_probability_chart(
            peak_day_df,
            label_column="day",
            title="Peak Probability by Day",
            x_label="Day",
            y_label="Peak Probability",
            rotate_x=False,
        )
        show_matplotlib_figure(fig)

        st.markdown("#### Department Peak Probability")
        st.dataframe(department_peak_df, use_container_width=True)
        fig = create_peak_probability_chart(
            department_peak_df,
            label_column="department",
            title="Department Peak Probability",
            x_label="Department",
            y_label="Peak Probability",
            rotate_x=True,
        )
        show_matplotlib_figure(fig)

        st.markdown("#### Department High Waiting Time Probability")
        st.dataframe(waiting_probability_dept_df, use_container_width=True)

    with tab3:
        st.markdown("### Poisson Probability and Appointment Behavior")

        booking_summary = calculate_walk_in_vs_booked_probability(appointment_df)
        booking_by_department_df = calculate_walk_in_vs_booked_probability_by_department(appointment_df)

        c1, c2 = st.columns(2)
        with c1:
            render_stat_card("Booked Probability", format_probability(booking_summary["booked_probability"]), "Proportion of booked patients")
        with c2:
            render_stat_card("Walk-in Probability", format_probability(booking_summary["walk_in_probability"]), "Proportion of walk-in patients")

        fig = create_booked_vs_walkin_chart(booking_summary)
        show_matplotlib_figure(fig)

        st.markdown("#### Walk-in vs Booked by Department")
        st.dataframe(booking_by_department_df, use_container_width=True)

        st.markdown("#### Poisson Probability Estimation")
        poisson_col1, poisson_col2, poisson_col3 = st.columns(3)

        with poisson_col1:
            poisson_scope = st.selectbox(
                "Poisson scope",
                options=["Overall Filtered OPD", "Specific Department"],
            )

        with poisson_col2:
            poisson_patient_count = st.number_input(
                "Patient count for probability",
                min_value=0,
                max_value=500,
                value=int(round(opd_df["patients_arrived"].mean())),
                step=1,
            )

        if poisson_scope == "Overall Filtered OPD":
            with poisson_col3:
                selected_day = st.selectbox(
                    "Day filter",
                    options=["All"] + sorted(opd_df["day"].dropna().unique().tolist()),
                )

            selected_hour_slot = st.selectbox(
                "Hour-slot filter",
                options=["All"] + sorted(opd_df["hour_slot"].dropna().unique().tolist()),
            )

            poisson_summary = get_filtered_poisson_probability_summary(
                opd_df=opd_df,
                patient_count=int(poisson_patient_count),
                selected_day=selected_day,
                selected_hour_slot=selected_hour_slot,
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Estimated Lambda",
                            "Patient Count",
                            "P(X = k)",
                            "P(X ≤ k)",
                            "P(X ≥ k)",
                        ],
                        "Value": [
                            poisson_summary["lambda_value"],
                            poisson_summary["patient_count"],
                            poisson_summary["probability_exact"],
                            poisson_summary["probability_at_most"],
                            poisson_summary["probability_at_least"],
                        ],
                    }
                ),
                use_container_width=True,
            )

        else:
            with poisson_col3:
                selected_department = st.selectbox(
                    "Department for Poisson estimation",
                    options=sorted(department_df["department"].dropna().unique().tolist()),
                )

            poisson_summary = get_department_poisson_probability_summary(
                department_df=department_df,
                department_name=selected_department,
                patient_count=int(poisson_patient_count),
            )

            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Department",
                            "Estimated Lambda",
                            "Patient Count",
                            "P(X = k)",
                            "P(X ≤ k)",
                            "P(X ≥ k)",
                        ],
                        "Value": [
                            poisson_summary["department"],
                            poisson_summary["lambda_value"],
                            poisson_summary["patient_count"],
                            poisson_summary["probability_exact"],
                            poisson_summary["probability_at_most"],
                            poisson_summary["probability_at_least"],
                        ],
                    }
                ),
                use_container_width=True,
            )


def render_simulation_section(opd_df: pd.DataFrame) -> None:
    render_section_header(
        "Simulation",
        "Model queue build-up, test what-if scenarios, and explore Monte Carlo behavior when arrivals wobble instead of marching in neat lines.",
    )

    available_days = get_available_days(opd_df)

    tab1, tab2, tab3 = st.tabs(
        [
            "Queue Simulation",
            "Scenario Comparison",
            "Monte Carlo Simulation",
        ]
    )

    with tab1:
        st.markdown("### Queue Simulation for a Selected Day")

        col1, col2 = st.columns(2)
        with col1:
            selected_day = st.selectbox("Select day", options=available_days, key="sim_day")
        with col2:
            opening_queue = st.number_input(
                "Opening queue",
                min_value=0,
                max_value=500,
                value=0,
                step=1,
                key="sim_opening_queue",
            )

        simulation_df = simulate_queue_for_day(opd_df, selected_day, opening_queue=opening_queue)
        simulation_summary = get_simulation_summary(simulation_df)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_stat_card("Total Arrivals", str(simulation_summary["total_arrivals"]), selected_day)
        with m2:
            render_stat_card("Total Served", str(simulation_summary["total_served"]), "Patients processed")
        with m3:
            render_stat_card("Final Queue", str(simulation_summary["final_queue"]), "Left at end of day")
        with m4:
            render_stat_card("Average Waiting", f"{simulation_summary['average_waiting_time']:.2f} min", "Simulation estimate")

        st.dataframe(simulation_df, use_container_width=True)

        fig = create_simulation_queue_chart(simulation_df)
        show_matplotlib_figure(fig)

        fig = create_simulation_waiting_time_chart(simulation_df)
        show_matplotlib_figure(fig)

    with tab2:
        st.markdown("### Scenario Comparison")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scenario_day = st.selectbox("Select day", options=available_days, key="scenario_day")
        with col2:
            scenario_opening_queue = st.number_input(
                "Opening queue",
                min_value=0,
                max_value=500,
                value=0,
                step=1,
                key="scenario_opening_queue",
            )
        with col3:
            arrival_multiplier = st.slider(
                "Arrival multiplier",
                min_value=0.50,
                max_value=2.00,
                value=1.15,
                step=0.05,
            )
        with col4:
            capacity_multiplier = st.slider(
                "Capacity multiplier",
                min_value=0.50,
                max_value=2.00,
                value=1.00,
                step=0.05,
            )

        comparison_result = compare_scenarios(
            opd_df=opd_df,
            selected_day=scenario_day,
            opening_queue=scenario_opening_queue,
            arrival_multiplier=arrival_multiplier,
            capacity_multiplier=capacity_multiplier,
        )

        st.dataframe(comparison_result["comparison_table"], use_container_width=True)

        fig = create_scenario_comparison_chart(comparison_result["comparison_table"])
        show_matplotlib_figure(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Baseline Summary")
            st.json(comparison_result["baseline_summary"])
        with c2:
            st.markdown("#### Scenario Summary")
            st.json(comparison_result["scenario_summary"])

    with tab3:
        st.markdown("### Monte Carlo Simulation")

        col1, col2, col3 = st.columns(3)
        with col1:
            monte_day = st.selectbox("Select day", options=available_days, key="monte_day")
        with col2:
            monte_opening_queue = st.number_input(
                "Opening queue",
                min_value=0,
                max_value=500,
                value=0,
                step=1,
                key="monte_opening_queue",
            )
        with col3:
            num_simulations = st.slider(
                "Number of simulations",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
            )

        monte_result = monte_carlo_simulation(
            opd_df=opd_df,
            selected_day=monte_day,
            opening_queue=monte_opening_queue,
            num_simulations=num_simulations,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            render_stat_card(
                "Avg Final Queue",
                str(monte_result["average_summary"]["average_final_queue"]),
                "Across all simulations",
            )
        with c2:
            render_stat_card(
                "Avg Waiting Time",
                f"{monte_result['average_summary']['average_waiting_time']:.2f} min",
                "Monte Carlo mean",
            )
        with c3:
            render_stat_card(
                "P(Final Queue > 20)",
                format_probability(monte_result["risk_summary"]["probability_final_queue_above_20"]),
                "Queue stress probability",
            )

        st.markdown("#### Risk Summary")
        st.json(monte_result["risk_summary"])

        st.markdown("#### Average Summary")
        st.json(monte_result["average_summary"])

        st.markdown("#### Raw Monte Carlo Results")
        st.dataframe(monte_result["monte_carlo_results"], use_container_width=True, height=320)

        fig = create_monte_carlo_histogram(
            monte_result["monte_carlo_results"],
            column_name="average_waiting_time",
            title="Monte Carlo Distribution of Average Waiting Time",
            x_label="Average Waiting Time (minutes)",
        )
        show_matplotlib_figure(fig)

        fig = create_monte_carlo_histogram(
            monte_result["monte_carlo_results"],
            column_name="final_queue",
            title="Monte Carlo Distribution of Final Queue",
            x_label="Final Queue",
        )
        show_matplotlib_figure(fig)


def generate_report_text(
    opd_df: pd.DataFrame,
    department_df: pd.DataFrame,
    appointment_df: pd.DataFrame,
) -> str:
    total_patients = int(pd.to_numeric(opd_df["patients_arrived"], errors="coerce").fillna(0).sum())
    avg_wait = float(pd.to_numeric(opd_df["avg_waiting_time"], errors="coerce").fillna(0).mean())
    avg_load_ratio = float(pd.to_numeric(opd_df["load_ratio"], errors="coerce").fillna(0).mean())

    peak_summary = get_peak_time_summary(opd_df)
    wait_summary = get_waiting_time_summary(opd_df)
    peak_hour_df = calculate_peak_hour_probability_by_hour(opd_df)
    peak_day_df = calculate_peak_day_probability(opd_df)
    department_peak_df = calculate_department_peak_probability(department_df)
    booking_summary = calculate_walk_in_vs_booked_probability(appointment_df)
    department_analysis_df = get_department_analysis(department_df)

    top_department = department_analysis_df.iloc[0]["department"] if not department_analysis_df.empty else "N/A"
    top_peak_department = department_peak_df.iloc[0]["department"] if not department_peak_df.empty else "N/A"

    report_text = f"""
Project Title:
Statistical and Probability-Based Analysis of OPD Patient Arrival Patterns for Better Appointment Planning and Queue Management

Overview:
This project analyzes synthetic but realistic OPD hospital data using statistics, probability estimation, prediction, and queue simulation. The main goal is to support better appointment planning and queue management in hospital OPD environments.

Key Findings:
1. The dataset recorded a total of {total_patients} patients across the observation period.
2. The average OPD waiting time was {avg_wait:.2f} minutes.
3. The average OPD load ratio was {avg_load_ratio:.2f}.
4. The busiest hour by patient volume was {peak_summary['busiest_hour']}.
5. The busiest day by patient volume was {peak_summary['busiest_day']}.
6. The most likely peak hour by probability was {top_peak_hour(peak_hour_df)}.
7. The most likely peak day by probability was {top_peak_day(peak_day_df)}.
8. The department with the highest total patient burden was {top_department}.
9. The department with the strongest peak tendency was {top_peak_department}.
10. The booked patient probability was {format_probability(booking_summary['booked_probability'])}, while the walk-in probability was {format_probability(booking_summary['walk_in_probability'])}.
11. The high waiting time rate was {format_probability(wait_summary['high_waiting_time_rate'])}.

Conclusion:
The results clearly show that OPD demand changes across hours, weekdays, and departments. These shifts create uneven waiting time and overload pressure. Statistical analysis, probability modelling, and simulation together provide a practical framework for better OPD planning.

Recommendation:
Hospitals should increase staffing or improve appointment balancing during peak time windows, watch high-pressure departments more carefully, and use demand probabilities to plan queues more effectively.
""".strip()

    return report_text


def render_report_insights_section(
    opd_df: pd.DataFrame,
    department_df: pd.DataFrame,
    appointment_df: pd.DataFrame,
) -> None:
    render_section_header(
        "Report Insights",
        "A neat report-friendly wrap-up so your presentation sounds sharp instead of accidentally reading like raw console output.",
    )

    report_text = generate_report_text(opd_df, department_df, appointment_df)
    peak_hour_df = calculate_peak_hour_probability_by_hour(opd_df)
    peak_day_df = calculate_peak_day_probability(opd_df)
    department_peak_df = calculate_department_peak_probability(department_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_stat_card("Most Likely Peak Hour", top_peak_hour(peak_hour_df), "Based on peak probability")
    with c2:
        render_stat_card("Most Likely Peak Day", top_peak_day(peak_day_df), "Based on peak probability")
    with c3:
        top_dept = department_peak_df.iloc[0]["department"] if not department_peak_df.empty else "N/A"
        render_stat_card("Most Likely Peak Department", top_dept, "Department peak tendency")

    st.text_area(
        "Report-ready summary",
        value=report_text,
        height=420,
    )

    st.download_button(
        label="Download report summary as TXT",
        data=report_text,
        file_name="opd_report_summary.txt",
        mime="text/plain",
    )

    st.markdown("### Key Points")
    st.markdown(
        """
        - OPD demand is not evenly distributed across time.
        - Peak hours and specific weekdays create noticeable waiting pressure.
        - Department-level behavior is different and matters for staffing decisions.
        - Walk-ins can significantly disturb scheduled appointment flow.
        - Queue simulation helps test operating stress before real overload happens.
        - Monte Carlo analysis helps estimate uncertainty rather than assuming perfect daily behavior.
        """
    )


def render_sidebar(all_data: Dict[str, pd.DataFrame]) -> None:
    opd_df = all_data["opd_patient_data"]
    department_df = all_data["department_wise_opd_data"]

    st.sidebar.markdown("## 🧭 Navigation")
    st.sidebar.caption("Choose a section and explore the OPD system layer by layer.")

    min_date = opd_df["date"].min()
    max_date = opd_df["date"].max()
    date_text = (
        f"{min_date.date()} → {max_date.date()}"
        if pd.notna(min_date) and pd.notna(max_date)
        else "Available"
    )

    st.sidebar.markdown("### 📌 Data Snapshot")
    st.sidebar.write(f"**Date range:** {date_text}")
    st.sidebar.write(f"**OPD rows:** {len(opd_df):,}")
    st.sidebar.write(f"**Departments:** {department_df['department'].nunique()}")

    st.sidebar.markdown("### 💡 Tip")
    st.sidebar.info(
        "For demo day, show Home → Statistical Analysis → Probability and Prediction → Simulation. "
        "That flow looks the most impressive."
    )


def main() -> None:
    inject_custom_css()

    with st.spinner("Loading datasets and preparing dashboard..."):
        try:
            all_data = load_project_data()
        except Exception as exc:
            st.error("Failed to load datasets.")
            st.exception(exc)
            st.stop()

    render_sidebar(all_data)

    section = st.sidebar.radio(
        "Go to section",
        options=[
            "Home",
            "Dataset Overview",
            "Statistical Analysis",
            "Probability and Prediction",
            "Simulation",
            "Report Insights",
        ],
    )

    opd_df = all_data["opd_patient_data"]
    department_df = all_data["department_wise_opd_data"]
    appointment_df = all_data["appointment_data"]

    if section == "Home":
        render_home_section(opd_df, department_df, appointment_df)
    elif section == "Dataset Overview":
        render_dataset_overview_section(all_data)
    elif section == "Statistical Analysis":
        render_statistical_analysis_section(opd_df, department_df)
    elif section == "Probability and Prediction":
        render_probability_and_prediction_section(opd_df, department_df, appointment_df)
    elif section == "Simulation":
        render_simulation_section(opd_df)
    elif section == "Report Insights":
        render_report_insights_section(opd_df, department_df, appointment_df)

    st.markdown(
        '<div class="footer-note">Built with Streamlit for OPD analytics, queue intelligence, and Report analysis.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()