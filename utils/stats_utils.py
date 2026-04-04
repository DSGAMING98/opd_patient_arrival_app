from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    validate_dataframe(df)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_columns


def extract_hour_start(hour_slot: str) -> Optional[int]:
    try:
        if pd.isna(hour_slot):
            return None
        return int(str(hour_slot).split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return None


def add_hour_start_column(df: pd.DataFrame, hour_slot_column: str = "hour_slot") -> pd.DataFrame:
    validate_dataframe(df, [hour_slot_column])

    result_df = df.copy()
    result_df["hour_start"] = result_df[hour_slot_column].apply(extract_hour_start)
    return result_df


def calculate_descriptive_statistics(df: pd.DataFrame, column_name: str) -> Dict[str, float]:
    validate_dataframe(df, [column_name])

    series = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if series.empty:
        raise ValueError(f"No valid numeric values found in column: {column_name}")

    mode_series = series.mode()

    return {
        "count": int(series.count()),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "mode": round(float(mode_series.iloc[0]), 2) if not mode_series.empty else np.nan,
        "variance": round(float(series.var(ddof=1)), 2) if len(series) > 1 else 0.0,
        "standard_deviation": round(float(series.std(ddof=1)), 2) if len(series) > 1 else 0.0,
        "minimum": round(float(series.min()), 2),
        "maximum": round(float(series.max()), 2),
        "range": round(float(series.max() - series.min()), 2),
        "q1": round(float(series.quantile(0.25)), 2),
        "q2": round(float(series.quantile(0.50)), 2),
        "q3": round(float(series.quantile(0.75)), 2),
        "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 2),
    }


def get_summary_statistics_table(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    validate_dataframe(df)

    rows = []
    for column_name in columns:
        if column_name in df.columns:
            try:
                stats_dict = calculate_descriptive_statistics(df, column_name)
                stats_dict["column"] = column_name
                rows.append(stats_dict)
            except ValueError:
                continue

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    ordered_columns = [
        "column",
        "count",
        "mean",
        "median",
        "mode",
        "variance",
        "standard_deviation",
        "minimum",
        "maximum",
        "range",
        "q1",
        "q2",
        "q3",
        "iqr",
    ]
    return summary_df[ordered_columns]


def get_hourly_patient_analysis(opd_df: pd.DataFrame) -> pd.DataFrame:
    validate_dataframe(
        opd_df,
        ["hour_slot", "patients_arrived", "avg_waiting_time", "load_ratio", "peak_indicator"],
    )

    working_df = add_hour_start_column(opd_df)

    grouped_df = (
        working_df.groupby(["hour_slot", "hour_start"], as_index=False)
        .agg(
            total_patients=("patients_arrived", "sum"),
            average_patients=("patients_arrived", "mean"),
            average_waiting_time=("avg_waiting_time", "mean"),
            average_load_ratio=("load_ratio", "mean"),
            peak_occurrences=("peak_indicator", "sum"),
            records=("patients_arrived", "count"),
        )
        .sort_values(["hour_start", "hour_slot"])
        .reset_index(drop=True)
    )

    grouped_df["average_patients"] = grouped_df["average_patients"].round(2)
    grouped_df["average_waiting_time"] = grouped_df["average_waiting_time"].round(2)
    grouped_df["average_load_ratio"] = grouped_df["average_load_ratio"].round(2)

    return grouped_df.drop(columns=["hour_start"])


def get_daywise_patient_analysis(opd_df: pd.DataFrame) -> pd.DataFrame:
    validate_dataframe(
        opd_df,
        ["day", "patients_arrived", "avg_waiting_time", "load_ratio", "peak_indicator"],
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    grouped_df = (
        opd_df.groupby("day", as_index=False)
        .agg(
            total_patients=("patients_arrived", "sum"),
            average_patients=("patients_arrived", "mean"),
            average_waiting_time=("avg_waiting_time", "mean"),
            average_load_ratio=("load_ratio", "mean"),
            peak_occurrences=("peak_indicator", "sum"),
            records=("patients_arrived", "count"),
        )
        .reset_index(drop=True)
    )

    grouped_df["day"] = pd.Categorical(grouped_df["day"], categories=day_order, ordered=True)
    grouped_df = grouped_df.sort_values("day").reset_index(drop=True)

    grouped_df["average_patients"] = grouped_df["average_patients"].round(2)
    grouped_df["average_waiting_time"] = grouped_df["average_waiting_time"].round(2)
    grouped_df["average_load_ratio"] = grouped_df["average_load_ratio"].round(2)

    return grouped_df


def get_department_analysis(department_df: pd.DataFrame) -> pd.DataFrame:
    validate_dataframe(
        department_df,
        ["department", "patients_arrived", "avg_waiting_time", "load_ratio", "peak_indicator"],
    )

    grouped_df = (
        department_df.groupby("department", as_index=False)
        .agg(
            total_patients=("patients_arrived", "sum"),
            average_patients=("patients_arrived", "mean"),
            average_waiting_time=("avg_waiting_time", "mean"),
            average_load_ratio=("load_ratio", "mean"),
            peak_occurrences=("peak_indicator", "sum"),
            records=("patients_arrived", "count"),
        )
        .sort_values(["total_patients", "average_waiting_time"], ascending=[False, False])
        .reset_index(drop=True)
    )

    grouped_df["average_patients"] = grouped_df["average_patients"].round(2)
    grouped_df["average_waiting_time"] = grouped_df["average_waiting_time"].round(2)
    grouped_df["average_load_ratio"] = grouped_df["average_load_ratio"].round(2)

    return grouped_df


def get_grouped_metric_analysis(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    validate_dataframe(df, [group_column, value_column])

    working_df = df[[group_column, value_column]].copy()
    working_df[value_column] = pd.to_numeric(working_df[value_column], errors="coerce")
    working_df = working_df.dropna()

    if working_df.empty:
        return pd.DataFrame()

    grouped_df = (
        working_df.groupby(group_column, as_index=False)[value_column]
        .agg(["count", "mean", "median", "min", "max", "std", "sum"])
        .reset_index()
    )

    grouped_df.columns = [
        group_column,
        "count",
        "mean",
        "median",
        "minimum",
        "maximum",
        "standard_deviation",
        "total",
    ]

    for col in ["mean", "median", "minimum", "maximum", "standard_deviation", "total"]:
        grouped_df[col] = grouped_df[col].fillna(0).round(2)

    return grouped_df


def calculate_correlation(df: pd.DataFrame, x_column: str, y_column: str) -> float:
    validate_dataframe(df, [x_column, y_column])

    working_df = df[[x_column, y_column]].copy()
    working_df[x_column] = pd.to_numeric(working_df[x_column], errors="coerce")
    working_df[y_column] = pd.to_numeric(working_df[y_column], errors="coerce")
    working_df = working_df.dropna()

    if len(working_df) < 2:
        raise ValueError("At least two valid rows are required to calculate correlation.")

    correlation_value = working_df[x_column].corr(working_df[y_column])

    if pd.isna(correlation_value):
        return 0.0

    return round(float(correlation_value), 4)


def perform_linear_regression(df: pd.DataFrame, x_column: str, y_column: str) -> Dict[str, float]:
    validate_dataframe(df, [x_column, y_column])

    working_df = df[[x_column, y_column]].copy()
    working_df[x_column] = pd.to_numeric(working_df[x_column], errors="coerce")
    working_df[y_column] = pd.to_numeric(working_df[y_column], errors="coerce")
    working_df = working_df.dropna()

    if len(working_df) < 2:
        raise ValueError("At least two valid rows are required for linear regression.")

    x_values = working_df[x_column].to_numpy(dtype=float)
    y_values = working_df[y_column].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted_values = slope * x_values + intercept

    ss_total = np.sum((y_values - np.mean(y_values)) ** 2)
    ss_residual = np.sum((y_values - predicted_values) ** 2)

    if ss_total == 0:
        r_squared = 1.0
    else:
        r_squared = 1 - (ss_residual / ss_total)

    correlation_value = working_df[x_column].corr(working_df[y_column])

    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "correlation": round(float(correlation_value), 4) if not pd.isna(correlation_value) else 0.0,
        "r_squared": round(float(r_squared), 4),
    }


def predict_from_regression(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_value: float,
) -> float:
    regression_result = perform_linear_regression(df, x_column, y_column)
    predicted_value = (regression_result["slope"] * float(x_value)) + regression_result["intercept"]
    return round(float(predicted_value), 2)


def get_peak_time_summary(opd_df: pd.DataFrame) -> Dict[str, object]:
    validate_dataframe(opd_df, ["hour_slot", "day", "patients_arrived", "avg_waiting_time", "peak_indicator"])

    hourly_df = get_hourly_patient_analysis(opd_df)
    daywise_df = get_daywise_patient_analysis(opd_df)

    busiest_hour_row = hourly_df.sort_values(
        ["total_patients", "average_waiting_time"],
        ascending=[False, False],
    ).iloc[0]

    busiest_day_row = daywise_df.sort_values(
        ["total_patients", "average_waiting_time"],
        ascending=[False, False],
    ).iloc[0]

    peak_rate = float(opd_df["peak_indicator"].mean()) if len(opd_df) > 0 else 0.0

    return {
        "busiest_hour": str(busiest_hour_row["hour_slot"]),
        "busiest_hour_total_patients": int(busiest_hour_row["total_patients"]),
        "busiest_day": str(busiest_day_row["day"]),
        "busiest_day_total_patients": int(busiest_day_row["total_patients"]),
        "overall_peak_rate": round(peak_rate, 4),
    }


def get_waiting_time_summary(opd_df: pd.DataFrame) -> Dict[str, float]:
    validate_dataframe(opd_df, ["avg_waiting_time"])

    series = pd.to_numeric(opd_df["avg_waiting_time"], errors="coerce").dropna()

    if series.empty:
        raise ValueError("No valid waiting time values found.")

    return {
        "average_waiting_time": round(float(series.mean()), 2),
        "median_waiting_time": round(float(series.median()), 2),
        "maximum_waiting_time": round(float(series.max()), 2),
        "minimum_waiting_time": round(float(series.min()), 2),
        "high_waiting_time_rate": round(float((series >= 30).mean()), 4),
    }


if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "day": ["Monday", "Monday", "Tuesday", "Tuesday"],
            "hour_slot": ["08:00-09:00", "09:00-10:00", "08:00-09:00", "09:00-10:00"],
            "patients_arrived": [20, 35, 18, 30],
            "avg_waiting_time": [12, 28, 10, 25],
            "load_ratio": [0.55, 0.91, 0.48, 0.84],
            "peak_indicator": [0, 1, 0, 1],
        }
    )

    print(calculate_descriptive_statistics(sample_data, "patients_arrived"))
    print(get_hourly_patient_analysis(sample_data))
    print(get_daywise_patient_analysis(sample_data))
    print(perform_linear_regression(sample_data, "patients_arrived", "avg_waiting_time"))