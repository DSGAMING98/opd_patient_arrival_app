from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def validate_chart_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


def _finalize_chart(
    fig,
    ax,
    title: str,
    x_label: str,
    y_label: str,
    rotate_x: bool = False,
    grid: bool = True,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if grid:
        ax.grid(True, alpha=0.3)

    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    rotate_x: bool = False,
):
    validate_chart_dataframe(df, [x_column, y_column])

    plot_df = df[[x_column, y_column]].copy()
    plot_df = plot_df.dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df[x_column].astype(str), plot_df[y_column])
    _finalize_chart(fig, ax, title, x_label, y_label, rotate_x=rotate_x)

    return fig


def create_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    rotate_x: bool = False,
    marker: str = "o",
):
    validate_chart_dataframe(df, [x_column, y_column])

    plot_df = df[[x_column, y_column]].copy()
    plot_df = plot_df.dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df[x_column].astype(str), plot_df[y_column], marker=marker)
    _finalize_chart(fig, ax, title, x_label, y_label, rotate_x=rotate_x)

    return fig


def create_histogram(
    df: pd.DataFrame,
    column_name: str,
    title: str,
    x_label: str,
    y_label: str = "Frequency",
    bins: int = 15,
):
    validate_chart_dataframe(df, [column_name])

    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No valid numeric values found in column: {column_name}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(series, bins=bins)
    _finalize_chart(fig, ax, title, x_label, y_label)

    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
):
    validate_chart_dataframe(df, [x_column, y_column])

    plot_df = df[[x_column, y_column]].copy()
    plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        raise ValueError("No valid numeric data available for scatter plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(plot_df[x_column], plot_df[y_column])
    _finalize_chart(fig, ax, title, x_label, y_label)

    return fig


def create_scatter_plot_with_regression(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
):
    validate_chart_dataframe(df, [x_column, y_column])

    plot_df = df[[x_column, y_column]].copy()
    plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna()

    if len(plot_df) < 2:
        raise ValueError("At least two valid rows are required for regression plot.")

    x_values = plot_df[x_column].to_numpy(dtype=float)
    y_values = plot_df[y_column].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x_values, y_values, 1)
    regression_line = slope * x_values + intercept

    sorted_index = np.argsort(x_values)
    x_sorted = x_values[sorted_index]
    y_sorted = regression_line[sorted_index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_values, y_values)
    ax.plot(x_sorted, y_sorted)
    _finalize_chart(fig, ax, title, x_label, y_label)

    return fig


def create_hourly_analysis_chart(hourly_df: pd.DataFrame):
    validate_chart_dataframe(hourly_df, ["hour_slot", "average_patients"])

    return create_bar_chart(
        df=hourly_df,
        x_column="hour_slot",
        y_column="average_patients",
        title="Average Patients by Hour Slot",
        x_label="Hour Slot",
        y_label="Average Patients",
        rotate_x=True,
    )


def create_daywise_analysis_chart(daywise_df: pd.DataFrame):
    validate_chart_dataframe(daywise_df, ["day", "average_patients"])

    return create_bar_chart(
        df=daywise_df,
        x_column="day",
        y_column="average_patients",
        title="Average Patients by Day",
        x_label="Day",
        y_label="Average Patients",
        rotate_x=False,
    )


def create_department_analysis_chart(department_df: pd.DataFrame):
    validate_chart_dataframe(department_df, ["department", "total_patients"])

    return create_bar_chart(
        df=department_df,
        x_column="department",
        y_column="total_patients",
        title="Total Patients by Department",
        x_label="Department",
        y_label="Total Patients",
        rotate_x=True,
    )


def create_waiting_time_distribution_chart(opd_df: pd.DataFrame):
    validate_chart_dataframe(opd_df, ["avg_waiting_time"])

    return create_histogram(
        df=opd_df,
        column_name="avg_waiting_time",
        title="Distribution of Average Waiting Time",
        x_label="Average Waiting Time (minutes)",
        bins=20,
    )


def create_load_ratio_distribution_chart(opd_df: pd.DataFrame):
    validate_chart_dataframe(opd_df, ["load_ratio"])

    return create_histogram(
        df=opd_df,
        column_name="load_ratio",
        title="Distribution of Load Ratio",
        x_label="Load Ratio",
        bins=20,
    )


def create_peak_probability_chart(
    df: pd.DataFrame,
    label_column: str,
    probability_column: str = "peak_probability",
    title: str = "Peak Probability",
    x_label: str = "Category",
    y_label: str = "Probability",
    rotate_x: bool = True,
):
    validate_chart_dataframe(df, [label_column, probability_column])

    return create_bar_chart(
        df=df,
        x_column=label_column,
        y_column=probability_column,
        title=title,
        x_label=x_label,
        y_label=y_label,
        rotate_x=rotate_x,
    )


def create_booked_vs_walkin_chart(probability_summary: dict):
    required_keys = {
        "booked_probability",
        "walk_in_probability",
    }

    if not isinstance(probability_summary, dict):
        raise ValueError("Input must be a dictionary.")

    missing_keys = [key for key in required_keys if key not in probability_summary]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    labels = ["Booked", "Walk-in"]
    values = [
        probability_summary["booked_probability"],
        probability_summary["walk_in_probability"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    _finalize_chart(
        fig,
        ax,
        title="Booked vs Walk-in Patient Probability",
        x_label="Patient Type",
        y_label="Probability",
    )

    return fig


def create_simulation_queue_chart(simulation_df: pd.DataFrame):
    validate_chart_dataframe(simulation_df, ["hour_slot", "queue_at_end"])

    return create_line_chart(
        df=simulation_df,
        x_column="hour_slot",
        y_column="queue_at_end",
        title="Queue at End of Each Hour",
        x_label="Hour Slot",
        y_label="Queue Size",
        rotate_x=True,
        marker="o",
    )


def create_simulation_waiting_time_chart(simulation_df: pd.DataFrame):
    validate_chart_dataframe(simulation_df, ["hour_slot", "estimated_waiting_time"])

    return create_line_chart(
        df=simulation_df,
        x_column="hour_slot",
        y_column="estimated_waiting_time",
        title="Estimated Waiting Time During Simulation",
        x_label="Hour Slot",
        y_label="Estimated Waiting Time (minutes)",
        rotate_x=True,
        marker="o",
    )


def create_scenario_comparison_chart(comparison_df: pd.DataFrame):
    validate_chart_dataframe(comparison_df, ["metric", "baseline", "scenario"])

    plot_df = comparison_df.copy()

    x_positions = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_positions - width / 2, plot_df["baseline"], width=width, label="Baseline")
    ax.bar(x_positions + width / 2, plot_df["scenario"], width=width, label="Scenario")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["metric"], rotation=45, ha="right")
    ax.legend()

    _finalize_chart(
        fig,
        ax,
        title="Baseline vs Scenario Comparison",
        x_label="Metric",
        y_label="Value",
        rotate_x=False,
    )

    return fig


def create_monte_carlo_histogram(
    monte_carlo_df: pd.DataFrame,
    column_name: str,
    title: str,
    x_label: str,
):
    validate_chart_dataframe(monte_carlo_df, [column_name])

    return create_histogram(
        df=monte_carlo_df,
        column_name=column_name,
        title=title,
        x_label=x_label,
        bins=20,
    )


def create_correlation_regression_chart(opd_df: pd.DataFrame):
    validate_chart_dataframe(opd_df, ["patients_arrived", "avg_waiting_time"])

    return create_scatter_plot_with_regression(
        df=opd_df,
        x_column="patients_arrived",
        y_column="avg_waiting_time",
        title="Patient Count vs Waiting Time",
        x_label="Patients Arrived",
        y_label="Average Waiting Time (minutes)",
    )


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "hour_slot": ["08:00-09:00", "09:00-10:00", "10:00-11:00"],
            "average_patients": [22, 35, 30],
            "avg_waiting_time": [12, 24, 20],
            "patients_arrived": [20, 36, 31],
            "load_ratio": [0.6, 0.95, 0.82],
        }
    )

    fig1 = create_hourly_analysis_chart(sample_df)
    plt.close(fig1)

    fig2 = create_scatter_plot_with_regression(
        sample_df,
        "patients_arrived",
        "avg_waiting_time",
        "Test Regression",
        "Patients",
        "Waiting Time",
    )
    plt.close(fig2)

    print("chart_utils.py test completed successfully.")