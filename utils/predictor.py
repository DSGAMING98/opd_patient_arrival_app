from typing import Dict, Optional

import numpy as np
import pandas as pd


def validate_prediction_dataframe(
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


def _prepare_numeric_pair_dataframe(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    validate_prediction_dataframe(df, [x_column, y_column])

    working_df = df[[x_column, y_column]].copy()
    working_df[x_column] = pd.to_numeric(working_df[x_column], errors="coerce")
    working_df[y_column] = pd.to_numeric(working_df[y_column], errors="coerce")
    working_df = working_df.dropna()

    if len(working_df) < 2:
        raise ValueError(
            f"At least two valid rows are required for columns '{x_column}' and '{y_column}'."
        )

    return working_df


def fit_linear_model(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> Dict[str, float]:
    working_df = _prepare_numeric_pair_dataframe(df, x_column, y_column)

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

    correlation_value = np.corrcoef(x_values, y_values)[0, 1]
    if np.isnan(correlation_value):
        correlation_value = 0.0

    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "correlation": round(float(correlation_value), 4),
        "r_squared": round(float(r_squared), 4),
    }


def predict_value_with_linear_model(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    input_x: float,
    minimum_value: Optional[float] = None,
) -> float:
    model = fit_linear_model(df, x_column, y_column)
    predicted_y = (model["slope"] * float(input_x)) + model["intercept"]

    if minimum_value is not None:
        predicted_y = max(float(minimum_value), predicted_y)

    return round(float(predicted_y), 2)


def predict_waiting_time_from_patient_count(
    opd_df: pd.DataFrame,
    patient_count: float,
) -> Dict[str, float]:
    validate_prediction_dataframe(opd_df, ["patients_arrived", "avg_waiting_time"])

    predicted_waiting_time = predict_value_with_linear_model(
        df=opd_df,
        x_column="patients_arrived",
        y_column="avg_waiting_time",
        input_x=patient_count,
        minimum_value=0.0,
    )

    model = fit_linear_model(opd_df, "patients_arrived", "avg_waiting_time")

    return {
        "input_patient_count": round(float(patient_count), 2),
        "predicted_waiting_time": round(float(predicted_waiting_time), 2),
        "slope": model["slope"],
        "intercept": model["intercept"],
        "correlation": model["correlation"],
        "r_squared": model["r_squared"],
    }


def estimate_capacity_from_doctors(
    df: pd.DataFrame,
    doctors_available: int,
    doctors_column: str = "doctors_available",
    capacity_column: str = "capacity_per_hour",
) -> float:
    validate_prediction_dataframe(df, [doctors_column, capacity_column])

    working_df = df[[doctors_column, capacity_column]].copy()
    working_df[doctors_column] = pd.to_numeric(working_df[doctors_column], errors="coerce")
    working_df[capacity_column] = pd.to_numeric(working_df[capacity_column], errors="coerce")
    working_df = working_df.dropna()

    working_df = working_df[working_df[doctors_column] > 0]

    if working_df.empty:
        raise ValueError("No valid doctor-capacity records available for estimation.")

    working_df["capacity_per_doctor"] = (
        working_df[capacity_column] / working_df[doctors_column]
    )

    average_capacity_per_doctor = working_df["capacity_per_doctor"].mean()
    estimated_capacity = float(doctors_available) * float(average_capacity_per_doctor)

    return round(max(0.0, estimated_capacity), 2)


def predict_load_ratio(
    opd_df: pd.DataFrame,
    patient_count: float,
    estimated_capacity: float,
) -> float:
    validate_prediction_dataframe(opd_df, ["patients_arrived", "load_ratio"])

    if estimated_capacity <= 0:
        return 1.5

    predicted_load_ratio = float(patient_count) / float(estimated_capacity)
    return round(max(0.0, predicted_load_ratio), 2)


def classify_load_risk(load_ratio: float) -> str:
    if load_ratio < 0.70:
        return "Low"
    if load_ratio < 0.90:
        return "Moderate"
    if load_ratio < 1.10:
        return "High"
    return "Critical"


def classify_waiting_time_risk(waiting_time: float) -> str:
    if waiting_time < 15:
        return "Low"
    if waiting_time < 30:
        return "Moderate"
    if waiting_time < 45:
        return "High"
    return "Critical"


def calculate_overload_probability_from_history(
    opd_df: pd.DataFrame,
    threshold: float = 0.90,
) -> float:
    validate_prediction_dataframe(opd_df, ["load_ratio"])

    series = pd.to_numeric(opd_df["load_ratio"], errors="coerce").dropna()
    if series.empty:
        return 0.0

    probability = float((series >= threshold).mean())
    return round(probability, 4)


def predict_overload_risk(
    opd_df: pd.DataFrame,
    patient_count: float,
    doctors_available: Optional[int] = None,
) -> Dict[str, float]:
    validate_prediction_dataframe(
        opd_df,
        ["patients_arrived", "avg_waiting_time", "doctors_available", "capacity_per_hour", "load_ratio"],
    )

    working_df = opd_df.copy()

    if doctors_available is None:
        doctors_available = int(round(pd.to_numeric(working_df["doctors_available"], errors="coerce").dropna().mean()))
        doctors_available = max(doctors_available, 1)

    estimated_capacity = estimate_capacity_from_doctors(
        df=working_df,
        doctors_available=doctors_available,
        doctors_column="doctors_available",
        capacity_column="capacity_per_hour",
    )

    predicted_waiting = predict_waiting_time_from_patient_count(
        opd_df=working_df,
        patient_count=patient_count,
    )["predicted_waiting_time"]

    predicted_load = predict_load_ratio(
        opd_df=working_df,
        patient_count=patient_count,
        estimated_capacity=estimated_capacity,
    )

    historical_overload_probability = calculate_overload_probability_from_history(
        opd_df=working_df,
        threshold=0.90,
    )

    load_risk = classify_load_risk(predicted_load)
    waiting_risk = classify_waiting_time_risk(predicted_waiting)

    if predicted_load >= 1.10 or predicted_waiting >= 45:
        overall_risk = "Critical"
    elif predicted_load >= 0.90 or predicted_waiting >= 30:
        overall_risk = "High"
    elif predicted_load >= 0.70 or predicted_waiting >= 15:
        overall_risk = "Moderate"
    else:
        overall_risk = "Low"

    return {
        "input_patient_count": round(float(patient_count), 2),
        "doctors_available": int(doctors_available),
        "estimated_capacity": round(float(estimated_capacity), 2),
        "predicted_load_ratio": round(float(predicted_load), 2),
        "predicted_waiting_time": round(float(predicted_waiting), 2),
        "historical_overload_probability": round(float(historical_overload_probability), 4),
        "load_risk_label": load_risk,
        "waiting_risk_label": waiting_risk,
        "overall_risk_label": overall_risk,
    }


def predict_waiting_time_by_department(
    department_df: pd.DataFrame,
    department_name: str,
    patient_count: float,
) -> Dict[str, float]:
    validate_prediction_dataframe(
        department_df,
        ["department", "patients_arrived", "avg_waiting_time"],
    )

    filtered_df = department_df[department_df["department"] == department_name].copy()

    if filtered_df.empty:
        raise ValueError(f"No records found for department: {department_name}")

    result = predict_waiting_time_from_patient_count(
        opd_df=filtered_df,
        patient_count=patient_count,
    )
    result["department"] = department_name
    return result


def predict_overload_risk_by_department(
    department_df: pd.DataFrame,
    department_name: str,
    patient_count: float,
    doctors_available: Optional[int] = None,
) -> Dict[str, float]:
    validate_prediction_dataframe(
        department_df,
        ["department", "patients_arrived", "avg_waiting_time", "doctors_available", "capacity_per_hour", "load_ratio"],
    )

    filtered_df = department_df[department_df["department"] == department_name].copy()

    if filtered_df.empty:
        raise ValueError(f"No records found for department: {department_name}")

    result = predict_overload_risk(
        opd_df=filtered_df,
        patient_count=patient_count,
        doctors_available=doctors_available,
    )
    result["department"] = department_name
    return result


def get_prediction_reference_ranges(opd_df: pd.DataFrame) -> Dict[str, float]:
    validate_prediction_dataframe(
        opd_df,
        ["patients_arrived", "avg_waiting_time", "load_ratio", "doctors_available", "capacity_per_hour"],
    )

    numeric_df = opd_df[
        ["patients_arrived", "avg_waiting_time", "load_ratio", "doctors_available", "capacity_per_hour"]
    ].copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    numeric_df = numeric_df.dropna()

    if numeric_df.empty:
        raise ValueError("No valid numeric records available for reference ranges.")

    return {
        "min_patients": int(numeric_df["patients_arrived"].min()),
        "max_patients": int(numeric_df["patients_arrived"].max()),
        "average_patients": round(float(numeric_df["patients_arrived"].mean()), 2),
        "average_waiting_time": round(float(numeric_df["avg_waiting_time"].mean()), 2),
        "average_load_ratio": round(float(numeric_df["load_ratio"].mean()), 2),
        "average_doctors_available": round(float(numeric_df["doctors_available"].mean()), 2),
        "average_capacity_per_hour": round(float(numeric_df["capacity_per_hour"].mean()), 2),
    }


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "patients_arrived": [20, 30, 40, 50, 60],
            "avg_waiting_time": [10, 15, 24, 34, 48],
            "doctors_available": [3, 3, 4, 4, 5],
            "capacity_per_hour": [24, 24, 30, 30, 36],
            "load_ratio": [0.83, 1.00, 1.00, 1.11, 1.25],
            "department": ["General Medicine"] * 5,
        }
    )

    print(predict_waiting_time_from_patient_count(sample_df, 45))
    print(predict_overload_risk(sample_df, 45, doctors_available=4))
    print(predict_waiting_time_by_department(sample_df, "General Medicine", 42))