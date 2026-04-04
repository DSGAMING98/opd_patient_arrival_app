from math import exp, factorial
from typing import Dict, Optional

import pandas as pd


def validate_probability_dataframe(
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


def _to_numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    validate_probability_dataframe(df, [column_name])
    series = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if series.empty:
        raise ValueError(f"No valid numeric values found in column: {column_name}")

    return series


def calculate_probability_from_condition(
    df: pd.DataFrame,
    condition_series: pd.Series,
) -> float:
    validate_probability_dataframe(df)

    if len(condition_series) != len(df):
        raise ValueError("Condition series length must match the DataFrame length.")

    probability = float(condition_series.fillna(False).mean())
    return round(probability, 4)


def calculate_overcrowding_probability(
    opd_df: pd.DataFrame,
    load_ratio_threshold: float = 0.90,
) -> float:
    validate_probability_dataframe(opd_df, ["load_ratio"])

    load_series = _to_numeric_series(opd_df, "load_ratio")
    probability = float((load_series >= load_ratio_threshold).mean())
    return round(probability, 4)


def calculate_high_waiting_time_probability(
    opd_df: pd.DataFrame,
    waiting_time_threshold: float = 30.0,
) -> float:
    validate_probability_dataframe(opd_df, ["avg_waiting_time"])

    wait_series = _to_numeric_series(opd_df, "avg_waiting_time")
    probability = float((wait_series >= waiting_time_threshold).mean())
    return round(probability, 4)


def calculate_peak_hour_probability_by_hour(opd_df: pd.DataFrame) -> pd.DataFrame:
    validate_probability_dataframe(opd_df, ["hour_slot", "peak_indicator", "patients_arrived", "avg_waiting_time"])

    working_df = opd_df.copy()
    working_df["peak_indicator"] = pd.to_numeric(working_df["peak_indicator"], errors="coerce").fillna(0)
    working_df["patients_arrived"] = pd.to_numeric(working_df["patients_arrived"], errors="coerce")
    working_df["avg_waiting_time"] = pd.to_numeric(working_df["avg_waiting_time"], errors="coerce")

    grouped_df = (
        working_df.groupby("hour_slot", as_index=False)
        .agg(
            total_records=("peak_indicator", "count"),
            peak_records=("peak_indicator", "sum"),
            average_patients=("patients_arrived", "mean"),
            average_waiting_time=("avg_waiting_time", "mean"),
        )
        .reset_index(drop=True)
    )

    grouped_df["peak_probability"] = (
        grouped_df["peak_records"] / grouped_df["total_records"]
    ).round(4)
    grouped_df["average_patients"] = grouped_df["average_patients"].round(2)
    grouped_df["average_waiting_time"] = grouped_df["average_waiting_time"].round(2)

    def hour_sort_key(slot: str) -> int:
        try:
            return int(str(slot).split(":")[0])
        except (ValueError, IndexError, AttributeError):
            return 999

    grouped_df["hour_order"] = grouped_df["hour_slot"].apply(hour_sort_key)
    grouped_df = grouped_df.sort_values(["hour_order", "hour_slot"]).drop(columns=["hour_order"]).reset_index(drop=True)

    return grouped_df


def calculate_peak_day_probability(opd_df: pd.DataFrame) -> pd.DataFrame:
    validate_probability_dataframe(opd_df, ["day", "peak_indicator", "patients_arrived", "avg_waiting_time"])

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    working_df = opd_df.copy()
    working_df["peak_indicator"] = pd.to_numeric(working_df["peak_indicator"], errors="coerce").fillna(0)
    working_df["patients_arrived"] = pd.to_numeric(working_df["patients_arrived"], errors="coerce")
    working_df["avg_waiting_time"] = pd.to_numeric(working_df["avg_waiting_time"], errors="coerce")

    grouped_df = (
        working_df.groupby("day", as_index=False)
        .agg(
            total_records=("peak_indicator", "count"),
            peak_records=("peak_indicator", "sum"),
            average_patients=("patients_arrived", "mean"),
            average_waiting_time=("avg_waiting_time", "mean"),
        )
        .reset_index(drop=True)
    )

    grouped_df["peak_probability"] = (
        grouped_df["peak_records"] / grouped_df["total_records"]
    ).round(4)
    grouped_df["average_patients"] = grouped_df["average_patients"].round(2)
    grouped_df["average_waiting_time"] = grouped_df["average_waiting_time"].round(2)

    grouped_df["day"] = pd.Categorical(grouped_df["day"], categories=day_order, ordered=True)
    grouped_df = grouped_df.sort_values("day").reset_index(drop=True)

    return grouped_df


def calculate_department_peak_probability(department_df: pd.DataFrame) -> pd.DataFrame:
    validate_probability_dataframe(
        department_df,
        ["department", "peak_indicator", "patients_arrived", "avg_waiting_time"],
    )

    working_df = department_df.copy()
    working_df["peak_indicator"] = pd.to_numeric(working_df["peak_indicator"], errors="coerce").fillna(0)
    working_df["patients_arrived"] = pd.to_numeric(working_df["patients_arrived"], errors="coerce")
    working_df["avg_waiting_time"] = pd.to_numeric(working_df["avg_waiting_time"], errors="coerce")

    grouped_df = (
        working_df.groupby("department", as_index=False)
        .agg(
            total_records=("peak_indicator", "count"),
            peak_records=("peak_indicator", "sum"),
            average_patients=("patients_arrived", "mean"),
            average_waiting_time=("avg_waiting_time", "mean"),
        )
        .reset_index(drop=True)
    )

    grouped_df["peak_probability"] = (
        grouped_df["peak_records"] / grouped_df["total_records"]
    ).round(4)
    grouped_df["average_patients"] = grouped_df["average_patients"].round(2)
    grouped_df["average_waiting_time"] = grouped_df["average_waiting_time"].round(2)

    grouped_df = grouped_df.sort_values(
        ["peak_probability", "average_waiting_time", "average_patients"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return grouped_df


def calculate_walk_in_vs_booked_probability(appointment_df: pd.DataFrame) -> Dict[str, float]:
    validate_probability_dataframe(
        appointment_df,
        ["booked_appointments", "walk_in_patients", "actual_patients"],
    )

    working_df = appointment_df.copy()
    working_df["booked_appointments"] = pd.to_numeric(working_df["booked_appointments"], errors="coerce").fillna(0)
    working_df["walk_in_patients"] = pd.to_numeric(working_df["walk_in_patients"], errors="coerce").fillna(0)
    working_df["actual_patients"] = pd.to_numeric(working_df["actual_patients"], errors="coerce").fillna(0)

    total_booked = float(working_df["booked_appointments"].sum())
    total_walk_in = float(working_df["walk_in_patients"].sum())
    total_actual = float(working_df["actual_patients"].sum())

    if total_actual <= 0:
        return {
            "total_booked_appointments": 0.0,
            "total_walk_in_patients": 0.0,
            "booked_probability": 0.0,
            "walk_in_probability": 0.0,
        }

    booked_probability = total_booked / total_actual
    walk_in_probability = total_walk_in / total_actual

    return {
        "total_booked_appointments": round(total_booked, 2),
        "total_walk_in_patients": round(total_walk_in, 2),
        "booked_probability": round(booked_probability, 4),
        "walk_in_probability": round(walk_in_probability, 4),
    }


def calculate_walk_in_vs_booked_probability_by_department(
    appointment_df: pd.DataFrame,
) -> pd.DataFrame:
    validate_probability_dataframe(
        appointment_df,
        ["department", "booked_appointments", "walk_in_patients", "actual_patients"],
    )

    working_df = appointment_df.copy()
    for column in ["booked_appointments", "walk_in_patients", "actual_patients"]:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce").fillna(0)

    grouped_df = (
        working_df.groupby("department", as_index=False)
        .agg(
            total_booked=("booked_appointments", "sum"),
            total_walk_in=("walk_in_patients", "sum"),
            total_actual=("actual_patients", "sum"),
        )
        .reset_index(drop=True)
    )

    grouped_df["booked_probability"] = grouped_df.apply(
        lambda row: round(float(row["total_booked"] / row["total_actual"]), 4) if row["total_actual"] > 0 else 0.0,
        axis=1,
    )
    grouped_df["walk_in_probability"] = grouped_df.apply(
        lambda row: round(float(row["total_walk_in"] / row["total_actual"]), 4) if row["total_actual"] > 0 else 0.0,
        axis=1,
    )

    grouped_df = grouped_df.sort_values(
        ["walk_in_probability", "booked_probability"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return grouped_df


def estimate_poisson_lambda(
    df: pd.DataFrame,
    column_name: str = "patients_arrived",
) -> float:
    series = _to_numeric_series(df, column_name)
    lambda_value = float(series.mean())
    return round(lambda_value, 4)


def poisson_probability_exact(k: int, lambda_value: float) -> float:
    if k < 0:
        return 0.0
    if lambda_value < 0:
        raise ValueError("Lambda value must be non-negative.")

    probability = (exp(-lambda_value) * (lambda_value ** k)) / factorial(k)
    return round(float(probability), 6)


def poisson_probability_at_most(k: int, lambda_value: float) -> float:
    if k < 0:
        return 0.0
    if lambda_value < 0:
        raise ValueError("Lambda value must be non-negative.")

    probability = sum(poisson_probability_exact(i, lambda_value) for i in range(k + 1))
    return round(float(probability), 6)


def poisson_probability_at_least(k: int, lambda_value: float) -> float:
    if k <= 0:
        return 1.0
    if lambda_value < 0:
        raise ValueError("Lambda value must be non-negative.")

    probability = 1.0 - poisson_probability_at_most(k - 1, lambda_value)
    return round(float(probability), 6)


def get_poisson_probability_summary(
    opd_df: pd.DataFrame,
    patient_count: int,
) -> Dict[str, float]:
    validate_probability_dataframe(opd_df, ["patients_arrived"])

    lambda_value = estimate_poisson_lambda(opd_df, "patients_arrived")

    return {
        "lambda_value": round(float(lambda_value), 4),
        "patient_count": int(patient_count),
        "probability_exact": poisson_probability_exact(patient_count, lambda_value),
        "probability_at_most": poisson_probability_at_most(patient_count, lambda_value),
        "probability_at_least": poisson_probability_at_least(patient_count, lambda_value),
    }


def get_filtered_poisson_probability_summary(
    opd_df: pd.DataFrame,
    patient_count: int,
    selected_day: Optional[str] = None,
    selected_hour_slot: Optional[str] = None,
) -> Dict[str, float]:
    validate_probability_dataframe(opd_df, ["day", "hour_slot", "patients_arrived"])

    filtered_df = opd_df.copy()

    if selected_day and selected_day != "All":
        filtered_df = filtered_df[filtered_df["day"] == selected_day]

    if selected_hour_slot and selected_hour_slot != "All":
        filtered_df = filtered_df[filtered_df["hour_slot"] == selected_hour_slot]

    if filtered_df.empty:
        raise ValueError("No records found for the selected filters.")

    return get_poisson_probability_summary(filtered_df, patient_count)


def get_department_poisson_probability_summary(
    department_df: pd.DataFrame,
    department_name: str,
    patient_count: int,
) -> Dict[str, float]:
    validate_probability_dataframe(department_df, ["department", "patients_arrived"])

    filtered_df = department_df[department_df["department"] == department_name].copy()

    if filtered_df.empty:
        raise ValueError(f"No records found for department: {department_name}")

    summary = get_poisson_probability_summary(filtered_df, patient_count)
    summary["department"] = department_name
    return summary


def calculate_waiting_time_probability_by_department(
    department_df: pd.DataFrame,
    waiting_time_threshold: float = 30.0,
) -> pd.DataFrame:
    validate_probability_dataframe(department_df, ["department", "avg_waiting_time"])

    working_df = department_df.copy()
    working_df["avg_waiting_time"] = pd.to_numeric(working_df["avg_waiting_time"], errors="coerce")

    grouped_df = (
        working_df.groupby("department", as_index=False)
        .agg(
            total_records=("avg_waiting_time", "count"),
            average_waiting_time=("avg_waiting_time", "mean"),
            max_waiting_time=("avg_waiting_time", "max"),
        )
        .reset_index(drop=True)
    )

    exceedance_df = (
        working_df.assign(high_wait=working_df["avg_waiting_time"] >= waiting_time_threshold)
        .groupby("department", as_index=False)["high_wait"]
        .mean()
        .rename(columns={"high_wait": "high_wait_probability"})
    )

    result_df = grouped_df.merge(exceedance_df, on="department", how="left")
    result_df["average_waiting_time"] = result_df["average_waiting_time"].round(2)
    result_df["max_waiting_time"] = result_df["max_waiting_time"].round(2)
    result_df["high_wait_probability"] = result_df["high_wait_probability"].fillna(0).round(4)

    result_df = result_df.sort_values(
        ["high_wait_probability", "average_waiting_time"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result_df


def get_probability_insight_summary(
    opd_df: pd.DataFrame,
    department_df: pd.DataFrame,
    appointment_df: pd.DataFrame,
) -> Dict[str, object]:
    overcrowding_probability = calculate_overcrowding_probability(opd_df)
    high_wait_probability = calculate_high_waiting_time_probability(opd_df)

    peak_hour_df = calculate_peak_hour_probability_by_hour(opd_df)
    peak_day_df = calculate_peak_day_probability(opd_df)
    dept_peak_df = calculate_department_peak_probability(department_df)
    booking_summary = calculate_walk_in_vs_booked_probability(appointment_df)

    most_peak_hour = peak_hour_df.iloc[0]["hour_slot"] if not peak_hour_df.empty else "N/A"
    most_peak_day = peak_day_df.sort_values("peak_probability", ascending=False).iloc[0]["day"] if not peak_day_df.empty else "N/A"
    most_peak_department = dept_peak_df.iloc[0]["department"] if not dept_peak_df.empty else "N/A"

    return {
        "overcrowding_probability": overcrowding_probability,
        "high_waiting_time_probability": high_wait_probability,
        "most_likely_peak_hour": most_peak_hour,
        "most_likely_peak_day": most_peak_day,
        "most_likely_peak_department": most_peak_department,
        "booked_probability": booking_summary["booked_probability"],
        "walk_in_probability": booking_summary["walk_in_probability"],
    }


if __name__ == "__main__":
    sample_opd_df = pd.DataFrame(
        {
            "day": ["Monday", "Monday", "Tuesday", "Tuesday", "Wednesday"],
            "hour_slot": ["08:00-09:00", "09:00-10:00", "08:00-09:00", "09:00-10:00", "10:00-11:00"],
            "patients_arrived": [20, 35, 18, 30, 40],
            "avg_waiting_time": [12, 28, 10, 25, 36],
            "load_ratio": [0.55, 0.91, 0.48, 0.84, 1.02],
            "peak_indicator": [0, 1, 0, 0, 1],
        }
    )

    sample_department_df = pd.DataFrame(
        {
            "department": ["General Medicine", "General Medicine", "Cardiology", "Cardiology"],
            "patients_arrived": [18, 24, 10, 15],
            "avg_waiting_time": [20, 31, 18, 40],
            "peak_indicator": [0, 1, 0, 1],
        }
    )

    sample_appointment_df = pd.DataFrame(
        {
            "department": ["General Medicine", "Cardiology"],
            "booked_appointments": [50, 20],
            "walk_in_patients": [25, 10],
            "actual_patients": [75, 30],
        }
    )

    print(calculate_overcrowding_probability(sample_opd_df))
    print(calculate_high_waiting_time_probability(sample_opd_df))
    print(calculate_peak_hour_probability_by_hour(sample_opd_df))
    print(get_poisson_probability_summary(sample_opd_df, 25))
    print(calculate_walk_in_vs_booked_probability(sample_appointment_df))