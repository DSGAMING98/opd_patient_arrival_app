from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
HOUR_STARTS = list(range(8, 17))  # 08:00 to 16:00


DEPARTMENT_CONFIG = {
    "General Medicine": {
        "weight": 0.30,
        "avg_consultation_time": 10,
        "doctor_count_weekday": (4, 5),
        "doctor_count_weekend": (2, 3),
        "max_patients_per_hour": (5, 7),
        "booking_rate": 0.45,
        "category_probs": [0.38, 0.34, 0.08, 0.20],  # New, Follow-up, Emergency, Senior Citizen
    },
    "Pediatrics": {
        "weight": 0.16,
        "avg_consultation_time": 12,
        "doctor_count_weekday": (2, 3),
        "doctor_count_weekend": (1, 2),
        "max_patients_per_hour": (4, 6),
        "booking_rate": 0.55,
        "category_probs": [0.45, 0.35, 0.05, 0.15],
    },
    "Orthopedics": {
        "weight": 0.14,
        "avg_consultation_time": 15,
        "doctor_count_weekday": (2, 3),
        "doctor_count_weekend": (1, 2),
        "max_patients_per_hour": (3, 5),
        "booking_rate": 0.50,
        "category_probs": [0.30, 0.42, 0.08, 0.20],
    },
    "Cardiology": {
        "weight": 0.12,
        "avg_consultation_time": 18,
        "doctor_count_weekday": (2, 3),
        "doctor_count_weekend": (1, 2),
        "max_patients_per_hour": (3, 4),
        "booking_rate": 0.72,
        "category_probs": [0.24, 0.48, 0.08, 0.20],
    },
    "Dermatology": {
        "weight": 0.10,
        "avg_consultation_time": 8,
        "doctor_count_weekday": (2, 3),
        "doctor_count_weekend": (1, 2),
        "max_patients_per_hour": (5, 7),
        "booking_rate": 0.68,
        "category_probs": [0.40, 0.40, 0.03, 0.17],
    },
    "ENT": {
        "weight": 0.10,
        "avg_consultation_time": 11,
        "doctor_count_weekday": (2, 3),
        "doctor_count_weekend": (1, 2),
        "max_patients_per_hour": (4, 6),
        "booking_rate": 0.58,
        "category_probs": [0.34, 0.40, 0.06, 0.20],
    },
    "Gynecology": {
        "weight": 0.08,
        "avg_consultation_time": 14,
        "doctor_count_weekday": (2, 3),
        "doctor_count_weekend": (1, 2),
        "max_patients_per_hour": (3, 5),
        "booking_rate": 0.76,
        "category_probs": [0.32, 0.46, 0.04, 0.18],
    },
}


PATIENT_CATEGORIES = ["New", "Follow-up", "Emergency", "Senior Citizen"]

DOCTOR_NAME_POOL = {
    "General Medicine": ["Dr. Mehta", "Dr. Rao", "Dr. Sharma", "Dr. Patel", "Dr. Nair", "Dr. Joshi"],
    "Pediatrics": ["Dr. Iyer", "Dr. Kapoor", "Dr. Fernandes", "Dr. Bhat"],
    "Orthopedics": ["Dr. Reddy", "Dr. Kulkarni", "Dr. Singh", "Dr. Das"],
    "Cardiology": ["Dr. Arora", "Dr. Menon", "Dr. Khan", "Dr. Thomas"],
    "Dermatology": ["Dr. Gupta", "Dr. Bedi", "Dr. Saha", "Dr. Gill"],
    "ENT": ["Dr. Varma", "Dr. Naidu", "Dr. Banerjee", "Dr. Prasad"],
    "Gynecology": ["Dr. Lakshmi", "Dr. Seema", "Dr. Anita", "Dr. Rekha"],
}


def hour_slot_label(hour_start: int) -> str:
    return f"{hour_start:02d}:00-{hour_start + 1:02d}:00"


def is_weekend(day_name: str) -> bool:
    return day_name in {"Saturday", "Sunday"}


def parse_hour_from_time(time_text: str) -> int:
    return int(str(time_text).split(":")[0])


def get_date_frame(start_date: str = "2026-01-01", num_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")
    df = pd.DataFrame({"date": dates})
    df["day"] = df["date"].dt.day_name()
    return df


def generate_doctor_schedule(
    date_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for _, row in date_df.iterrows():
        current_date = row["date"]
        day_name = row["day"]

        for department, config in DEPARTMENT_CONFIG.items():
            if is_weekend(day_name):
                min_docs, max_docs = config["doctor_count_weekend"]
            else:
                min_docs, max_docs = config["doctor_count_weekday"]

            doctor_count = int(rng.integers(min_docs, max_docs + 1))
            available_names = DOCTOR_NAME_POOL[department]

            for i in range(doctor_count):
                doctor_name = available_names[i % len(available_names)]

                shift_start = int(rng.choice([8, 9]))
                shift_length = int(rng.choice([6, 7, 8]))
                shift_end = min(17, shift_start + shift_length)

                max_patients_per_hour = int(
                    rng.integers(
                        config["max_patients_per_hour"][0],
                        config["max_patients_per_hour"][1] + 1,
                    )
                )

                rows.append(
                    {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "day": day_name,
                        "department": department,
                        "doctor_name": doctor_name,
                        "shift_start": f"{shift_start:02d}:00",
                        "shift_end": f"{shift_end:02d}:00",
                        "max_patients_per_hour": max_patients_per_hour,
                    }
                )

    schedule_df = pd.DataFrame(rows)
    return schedule_df


def get_active_doctors(
    schedule_df: pd.DataFrame,
    date_text: str,
    department: str,
    hour_start: int,
) -> pd.DataFrame:
    dept_schedule = schedule_df[
        (schedule_df["date"] == date_text) & (schedule_df["department"] == department)
    ].copy()

    if dept_schedule.empty:
        return dept_schedule

    dept_schedule["shift_start_hour"] = dept_schedule["shift_start"].apply(parse_hour_from_time)
    dept_schedule["shift_end_hour"] = dept_schedule["shift_end"].apply(parse_hour_from_time)

    active_df = dept_schedule[
        (dept_schedule["shift_start_hour"] <= hour_start)
        & (dept_schedule["shift_end_hour"] > hour_start)
    ].copy()

    return active_df


def generate_department_wise_opd_data(
    date_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    hour_factor = {
        8: 0.65,
        9: 0.90,
        10: 1.20,
        11: 1.32,
        12: 1.12,
        13: 0.95,
        14: 0.88,
        15: 0.72,
        16: 0.54,
    }

    day_factor = {
        "Monday": 1.10,
        "Tuesday": 1.02,
        "Wednesday": 1.00,
        "Thursday": 1.04,
        "Friday": 1.08,
        "Saturday": 0.72,
        "Sunday": 0.48,
    }

    for date_index, row in date_df.reset_index(drop=True).iterrows():
        date_text = row["date"].strftime("%Y-%m-%d")
        day_name = row["day"]

        base_daily_patients = 125
        trend_factor = 1.0 + (date_index / max(len(date_df), 1)) * 0.10
        noise_day_factor = float(rng.normal(1.0, 0.07))
        adjusted_base = base_daily_patients * day_factor[day_name] * trend_factor * noise_day_factor

        for department, config in DEPARTMENT_CONFIG.items():
            dept_weight = config["weight"]
            base_consult_time = config["avg_consultation_time"]

            for hour_start in HOUR_STARTS:
                active_doctors_df = get_active_doctors(schedule_df, date_text, department, hour_start)
                doctors_available = int(len(active_doctors_df))

                if doctors_available > 0:
                    capacity_per_hour = int(active_doctors_df["max_patients_per_hour"].sum())
                else:
                    capacity_per_hour = 0

                expected_patients = adjusted_base * dept_weight * hour_factor[hour_start]
                expected_patients *= float(rng.normal(1.0, 0.10))
                expected_patients = max(expected_patients, 0.5)

                patients_arrived = int(rng.poisson(lam=expected_patients))

                if doctors_available == 0:
                    avg_consultation_time = float(base_consult_time)
                    avg_waiting_time = float(rng.uniform(20, 40))
                    load_ratio = 1.20 if patients_arrived > 0 else 0.0
                else:
                    avg_consultation_time = float(
                        np.clip(
                            rng.normal(base_consult_time, 1.5),
                            max(5, base_consult_time - 3),
                            base_consult_time + 4,
                        )
                    )

                    load_ratio = round(patients_arrived / max(capacity_per_hour, 1), 2)

                    base_wait = 5 + (avg_consultation_time * 0.6)
                    overload_component = max(0, load_ratio - 0.65) * 40
                    variability = float(rng.normal(0, 4))
                    avg_waiting_time = float(
                        np.clip(base_wait + overload_component + variability, 3, 120)
                    )

                peak_indicator = 1 if (load_ratio >= 0.85 or patients_arrived >= 15) else 0

                rows.append(
                    {
                        "date": date_text,
                        "day": day_name,
                        "hour_slot": hour_slot_label(hour_start),
                        "department": department,
                        "patients_arrived": int(patients_arrived),
                        "doctors_available": int(doctors_available),
                        "capacity_per_hour": int(capacity_per_hour),
                        "avg_consultation_time": round(avg_consultation_time, 2),
                        "avg_waiting_time": round(avg_waiting_time, 2),
                        "load_ratio": round(load_ratio, 2),
                        "peak_indicator": int(peak_indicator),
                    }
                )

    department_df = pd.DataFrame(rows)
    return department_df


def generate_main_opd_data(department_df: pd.DataFrame) -> pd.DataFrame:
    if department_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "day",
                "hour_slot",
                "patients_arrived",
                "doctors_available",
                "capacity_per_hour",
                "avg_consultation_time",
                "avg_waiting_time",
                "load_ratio",
                "peak_indicator",
            ]
        )

    grouped = (
        department_df.groupby(["date", "day", "hour_slot"], as_index=False)
        .agg(
            patients_arrived=("patients_arrived", "sum"),
            doctors_available=("doctors_available", "sum"),
            capacity_per_hour=("capacity_per_hour", "sum"),
        )
        .sort_values(["date", "hour_slot"])
        .reset_index(drop=True)
    )

    weighted_consult = []
    weighted_wait = []

    for _, row in grouped.iterrows():
        subset = department_df[
            (department_df["date"] == row["date"])
            & (department_df["day"] == row["day"])
            & (department_df["hour_slot"] == row["hour_slot"])
        ].copy()

        weights = subset["patients_arrived"].replace(0, 1)

        avg_consult = np.average(subset["avg_consultation_time"], weights=weights)
        avg_wait = np.average(subset["avg_waiting_time"], weights=weights)

        weighted_consult.append(round(float(avg_consult), 2))
        weighted_wait.append(round(float(avg_wait), 2))

    grouped["avg_consultation_time"] = weighted_consult
    grouped["avg_waiting_time"] = weighted_wait
    grouped["load_ratio"] = (
        grouped["patients_arrived"] / grouped["capacity_per_hour"].replace(0, np.nan)
    ).fillna(0).round(2)
    grouped["peak_indicator"] = ((grouped["load_ratio"] >= 0.85) | (grouped["patients_arrived"] >= 60)).astype(int)

    return grouped


def generate_appointment_data(
    department_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for _, row in department_df.iterrows():
        department = row["department"]
        actual_patients = int(row["patients_arrived"])
        booking_rate = DEPARTMENT_CONFIG[department]["booking_rate"]

        if actual_patients == 0:
            booked = 0
            walk_in = 0
            total_expected = 0
        else:
            booked = int(round(actual_patients * float(rng.normal(booking_rate, 0.08))))
            booked = int(np.clip(booked, 0, actual_patients))

            walk_in = max(actual_patients - booked, 0)

            expected_walk_in = int(
                max(
                    0,
                    round(walk_in * float(rng.normal(1.0, 0.12))),
                )
            )
            total_expected = booked + expected_walk_in

        rows.append(
            {
                "date": row["date"],
                "day": row["day"],
                "hour_slot": row["hour_slot"],
                "department": department,
                "booked_appointments": int(booked),
                "walk_in_patients": int(walk_in),
                "total_expected_patients": int(total_expected),
                "actual_patients": int(actual_patients),
            }
        )

    appointment_df = pd.DataFrame(rows)
    return appointment_df


def generate_patient_category_data(
    department_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for _, row in department_df.iterrows():
        department = row["department"]
        total_patients = int(row["patients_arrived"])

        base_probs = np.array(DEPARTMENT_CONFIG[department]["category_probs"], dtype=float)

        if "11:00" in row["hour_slot"] or "12:00" in row["hour_slot"]:
            base_probs[2] += 0.02  # Slight bump for Emergency around peak OPD time
            base_probs = base_probs / base_probs.sum()

        if total_patients > 0:
            category_counts = rng.multinomial(total_patients, base_probs)
        else:
            category_counts = np.array([0, 0, 0, 0])

        for category_name, count in zip(PATIENT_CATEGORIES, category_counts):
            rows.append(
                {
                    "date": row["date"],
                    "day": row["day"],
                    "hour_slot": row["hour_slot"],
                    "department": department,
                    "patient_category": category_name,
                    "count": int(count),
                }
            )

    category_df = pd.DataFrame(rows)
    return category_df


def save_dataframe(df: pd.DataFrame, file_name: str) -> None:
    file_path = DATA_DIR / file_name
    df.to_csv(file_path, index=False)


def generate_all_datasets(
    start_date: str = "2026-01-01",
    num_days: int = 60,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    date_df = get_date_frame(start_date=start_date, num_days=num_days)
    schedule_df = generate_doctor_schedule(date_df=date_df, rng=rng)
    department_df = generate_department_wise_opd_data(
        date_df=date_df,
        schedule_df=schedule_df,
        rng=rng,
    )
    opd_df = generate_main_opd_data(department_df=department_df)
    appointment_df = generate_appointment_data(department_df=department_df, rng=rng)
    category_df = generate_patient_category_data(department_df=department_df, rng=rng)

    save_dataframe(opd_df, "opd_patient_data.csv")
    save_dataframe(department_df, "department_wise_opd_data.csv")
    save_dataframe(schedule_df, "doctor_schedule_data.csv")
    save_dataframe(appointment_df, "appointment_data.csv")
    save_dataframe(category_df, "patient_category_data.csv")

    return {
        "opd_patient_data": opd_df,
        "department_wise_opd_data": department_df,
        "doctor_schedule_data": schedule_df,
        "appointment_data": appointment_df,
        "patient_category_data": category_df,
    }


if __name__ == "__main__":
    generated = generate_all_datasets()
    print("Synthetic OPD datasets generated successfully.")
    for name, df in generated.items():
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")