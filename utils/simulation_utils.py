from typing import Dict, Optional

import numpy as np
import pandas as pd


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def validate_simulation_dataframe(
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


def extract_hour_start(hour_slot: str) -> int:
    try:
        return int(str(hour_slot).split(":")[0])
    except (ValueError, IndexError, AttributeError):
        return 999


def prepare_day_data(opd_df: pd.DataFrame, selected_day: str) -> pd.DataFrame:
    validate_simulation_dataframe(
        opd_df,
        ["day", "hour_slot", "patients_arrived", "capacity_per_hour", "avg_consultation_time", "avg_waiting_time"],
    )

    filtered_df = opd_df[opd_df["day"] == selected_day].copy()

    if filtered_df.empty:
        raise ValueError(f"No records found for selected day: {selected_day}")

    numeric_columns = [
        "patients_arrived",
        "capacity_per_hour",
        "avg_consultation_time",
        "avg_waiting_time",
    ]

    for column in numeric_columns:
        filtered_df[column] = pd.to_numeric(filtered_df[column], errors="coerce")

    filtered_df = filtered_df.dropna(subset=numeric_columns)

    if filtered_df.empty:
        raise ValueError(f"No valid numeric records found for selected day: {selected_day}")

    grouped_df = (
        filtered_df.groupby(["day", "hour_slot"], as_index=False)
        .agg(
            patients_arrived=("patients_arrived", "mean"),
            capacity_per_hour=("capacity_per_hour", "mean"),
            avg_consultation_time=("avg_consultation_time", "mean"),
            avg_waiting_time=("avg_waiting_time", "mean"),
        )
        .reset_index(drop=True)
    )

    grouped_df["patients_arrived"] = grouped_df["patients_arrived"].round().astype(int)
    grouped_df["capacity_per_hour"] = grouped_df["capacity_per_hour"].round().astype(int)
    grouped_df["avg_consultation_time"] = grouped_df["avg_consultation_time"].round(2)
    grouped_df["avg_waiting_time"] = grouped_df["avg_waiting_time"].round(2)
    grouped_df["hour_order"] = grouped_df["hour_slot"].apply(extract_hour_start)

    grouped_df = grouped_df.sort_values(["hour_order", "hour_slot"]).reset_index(drop=True)
    return grouped_df.drop(columns=["hour_order"])


def run_queue_simulation(
    day_df: pd.DataFrame,
    opening_queue: int = 0,
) -> pd.DataFrame:
    validate_simulation_dataframe(
        day_df,
        ["day", "hour_slot", "patients_arrived", "capacity_per_hour", "avg_consultation_time", "avg_waiting_time"],
    )

    simulation_rows = []
    queue_carry_forward = max(0, int(opening_queue))

    for _, row in day_df.iterrows():
        arrivals = max(0, int(round(float(row["patients_arrived"]))))
        capacity = max(0, int(round(float(row["capacity_per_hour"]))))
        avg_consultation_time = max(1.0, float(row["avg_consultation_time"]))
        base_waiting_time = max(0.0, float(row["avg_waiting_time"]))

        total_demand = queue_carry_forward + arrivals
        served_patients = min(total_demand, capacity)
        queue_end = max(0, total_demand - served_patients)

        if capacity > 0:
            utilization_ratio = total_demand / capacity
            queue_pressure = (queue_carry_forward / capacity) * avg_consultation_time
            estimated_waiting_time = base_waiting_time + queue_pressure
        else:
            utilization_ratio = 0.0
            estimated_waiting_time = base_waiting_time + 30.0 if total_demand > 0 else base_waiting_time

        overload_flag = 1 if total_demand > capacity else 0

        simulation_rows.append(
            {
                "day": row["day"],
                "hour_slot": row["hour_slot"],
                "patients_arrived": arrivals,
                "capacity_per_hour": capacity,
                "queue_at_start": int(queue_carry_forward),
                "total_demand": int(total_demand),
                "served_patients": int(served_patients),
                "queue_at_end": int(queue_end),
                "utilization_ratio": round(float(utilization_ratio), 2),
                "estimated_waiting_time": round(float(estimated_waiting_time), 2),
                "overload_flag": int(overload_flag),
            }
        )

        queue_carry_forward = queue_end

    return pd.DataFrame(simulation_rows)


def simulate_queue_for_day(
    opd_df: pd.DataFrame,
    selected_day: str,
    opening_queue: int = 0,
) -> pd.DataFrame:
    day_df = prepare_day_data(opd_df, selected_day)
    return run_queue_simulation(day_df=day_df, opening_queue=opening_queue)


def get_simulation_summary(simulation_df: pd.DataFrame) -> Dict[str, float]:
    validate_simulation_dataframe(
        simulation_df,
        [
            "patients_arrived",
            "capacity_per_hour",
            "served_patients",
            "queue_at_end",
            "estimated_waiting_time",
            "overload_flag",
            "utilization_ratio",
        ],
    )

    total_arrivals = int(pd.to_numeric(simulation_df["patients_arrived"], errors="coerce").fillna(0).sum())
    total_capacity = int(pd.to_numeric(simulation_df["capacity_per_hour"], errors="coerce").fillna(0).sum())
    total_served = int(pd.to_numeric(simulation_df["served_patients"], errors="coerce").fillna(0).sum())
    final_queue = int(pd.to_numeric(simulation_df["queue_at_end"], errors="coerce").fillna(0).iloc[-1])
    average_wait = float(pd.to_numeric(simulation_df["estimated_waiting_time"], errors="coerce").fillna(0).mean())
    max_wait = float(pd.to_numeric(simulation_df["estimated_waiting_time"], errors="coerce").fillna(0).max())
    overload_hours = int(pd.to_numeric(simulation_df["overload_flag"], errors="coerce").fillna(0).sum())
    average_utilization = float(pd.to_numeric(simulation_df["utilization_ratio"], errors="coerce").fillna(0).mean())

    return {
        "total_arrivals": total_arrivals,
        "total_capacity": total_capacity,
        "total_served": total_served,
        "final_queue": final_queue,
        "average_waiting_time": round(average_wait, 2),
        "maximum_waiting_time": round(max_wait, 2),
        "overload_hours": overload_hours,
        "average_utilization_ratio": round(average_utilization, 2),
    }


def create_adjusted_day_data(
    day_df: pd.DataFrame,
    arrival_multiplier: float = 1.0,
    capacity_multiplier: float = 1.0,
) -> pd.DataFrame:
    validate_simulation_dataframe(
        day_df,
        ["patients_arrived", "capacity_per_hour", "avg_consultation_time", "avg_waiting_time"],
    )

    adjusted_df = day_df.copy()

    adjusted_df["patients_arrived"] = (
        pd.to_numeric(adjusted_df["patients_arrived"], errors="coerce").fillna(0) * float(arrival_multiplier)
    ).round().astype(int)

    adjusted_df["capacity_per_hour"] = (
        pd.to_numeric(adjusted_df["capacity_per_hour"], errors="coerce").fillna(0) * float(capacity_multiplier)
    ).round().astype(int)

    adjusted_df["capacity_per_hour"] = adjusted_df["capacity_per_hour"].clip(lower=0)
    adjusted_df["patients_arrived"] = adjusted_df["patients_arrived"].clip(lower=0)

    return adjusted_df


def compare_scenarios(
    opd_df: pd.DataFrame,
    selected_day: str,
    opening_queue: int = 0,
    arrival_multiplier: float = 1.15,
    capacity_multiplier: float = 1.00,
) -> Dict[str, object]:
    baseline_day_df = prepare_day_data(opd_df, selected_day)
    baseline_simulation_df = run_queue_simulation(
        day_df=baseline_day_df,
        opening_queue=opening_queue,
    )
    baseline_summary = get_simulation_summary(baseline_simulation_df)

    adjusted_day_df = create_adjusted_day_data(
        day_df=baseline_day_df,
        arrival_multiplier=arrival_multiplier,
        capacity_multiplier=capacity_multiplier,
    )
    scenario_simulation_df = run_queue_simulation(
        day_df=adjusted_day_df,
        opening_queue=opening_queue,
    )
    scenario_summary = get_simulation_summary(scenario_simulation_df)

    comparison_df = pd.DataFrame(
        {
            "metric": [
                "Total Arrivals",
                "Total Capacity",
                "Total Served",
                "Final Queue",
                "Average Waiting Time",
                "Maximum Waiting Time",
                "Overload Hours",
                "Average Utilization Ratio",
            ],
            "baseline": [
                baseline_summary["total_arrivals"],
                baseline_summary["total_capacity"],
                baseline_summary["total_served"],
                baseline_summary["final_queue"],
                baseline_summary["average_waiting_time"],
                baseline_summary["maximum_waiting_time"],
                baseline_summary["overload_hours"],
                baseline_summary["average_utilization_ratio"],
            ],
            "scenario": [
                scenario_summary["total_arrivals"],
                scenario_summary["total_capacity"],
                scenario_summary["total_served"],
                scenario_summary["final_queue"],
                scenario_summary["average_waiting_time"],
                scenario_summary["maximum_waiting_time"],
                scenario_summary["overload_hours"],
                scenario_summary["average_utilization_ratio"],
            ],
        }
    )

    comparison_df["difference"] = comparison_df["scenario"] - comparison_df["baseline"]

    return {
        "baseline_simulation": baseline_simulation_df,
        "scenario_simulation": scenario_simulation_df,
        "baseline_summary": baseline_summary,
        "scenario_summary": scenario_summary,
        "comparison_table": comparison_df,
    }


def monte_carlo_simulation(
    opd_df: pd.DataFrame,
    selected_day: str,
    opening_queue: int = 0,
    num_simulations: int = 300,
    random_seed: int = 42,
) -> Dict[str, object]:
    baseline_day_df = prepare_day_data(opd_df, selected_day)
    rng = np.random.default_rng(random_seed)

    summary_rows = []

    for simulation_id in range(num_simulations):
        simulated_day_df = baseline_day_df.copy()

        simulated_arrivals = []
        simulated_capacity = []

        for _, row in simulated_day_df.iterrows():
            base_arrivals = max(0, int(row["patients_arrived"]))
            base_capacity = max(0, int(row["capacity_per_hour"]))

            arrival_lambda = max(0.1, float(base_arrivals))
            new_arrivals = int(rng.poisson(arrival_lambda))

            capacity_noise = float(rng.normal(loc=1.0, scale=0.08))
            new_capacity = int(round(base_capacity * capacity_noise))
            new_capacity = max(0, new_capacity)

            simulated_arrivals.append(new_arrivals)
            simulated_capacity.append(new_capacity)

        simulated_day_df["patients_arrived"] = simulated_arrivals
        simulated_day_df["capacity_per_hour"] = simulated_capacity

        simulation_df = run_queue_simulation(
            day_df=simulated_day_df,
            opening_queue=opening_queue,
        )
        summary = get_simulation_summary(simulation_df)
        summary["simulation_id"] = simulation_id + 1
        summary_rows.append(summary)

    monte_carlo_df = pd.DataFrame(summary_rows)

    average_summary = {
        "average_total_arrivals": round(float(monte_carlo_df["total_arrivals"].mean()), 2),
        "average_total_served": round(float(monte_carlo_df["total_served"].mean()), 2),
        "average_final_queue": round(float(monte_carlo_df["final_queue"].mean()), 2),
        "average_waiting_time": round(float(monte_carlo_df["average_waiting_time"].mean()), 2),
        "average_maximum_waiting_time": round(float(monte_carlo_df["maximum_waiting_time"].mean()), 2),
        "average_overload_hours": round(float(monte_carlo_df["overload_hours"].mean()), 2),
        "average_utilization_ratio": round(float(monte_carlo_df["average_utilization_ratio"].mean()), 2),
    }

    risk_summary = {
        "probability_final_queue_above_20": round(float((monte_carlo_df["final_queue"] > 20).mean()), 4),
        "probability_average_wait_above_30": round(float((monte_carlo_df["average_waiting_time"] > 30).mean()), 4),
        "probability_overload_hours_at_least_3": round(float((monte_carlo_df["overload_hours"] >= 3).mean()), 4),
    }

    return {
        "monte_carlo_results": monte_carlo_df,
        "average_summary": average_summary,
        "risk_summary": risk_summary,
    }


def get_available_days(opd_df: pd.DataFrame) -> list:
    validate_simulation_dataframe(opd_df, ["day"])

    available_days = opd_df["day"].dropna().astype(str).unique().tolist()
    available_days = [day for day in DAY_ORDER if day in available_days]
    return available_days


if __name__ == "__main__":
    sample_df = pd.DataFrame(
        {
            "day": [
                "Monday", "Monday", "Monday", "Monday",
                "Tuesday", "Tuesday", "Tuesday", "Tuesday",
            ],
            "hour_slot": [
                "08:00-09:00", "09:00-10:00", "10:00-11:00", "11:00-12:00",
                "08:00-09:00", "09:00-10:00", "10:00-11:00", "11:00-12:00",
            ],
            "patients_arrived": [18, 28, 34, 24, 15, 22, 31, 20],
            "capacity_per_hour": [20, 22, 24, 22, 18, 20, 22, 20],
            "avg_consultation_time": [10, 11, 11, 10, 10, 10, 11, 10],
            "avg_waiting_time": [12, 18, 26, 20, 10, 14, 22, 16],
        }
    )

    monday_simulation = simulate_queue_for_day(sample_df, "Monday")
    print(monday_simulation)
    print(get_simulation_summary(monday_simulation))

    comparison = compare_scenarios(
        opd_df=sample_df,
        selected_day="Monday",
        arrival_multiplier=1.20,
        capacity_multiplier=1.05,
    )
    print(comparison["comparison_table"])

    mc_result = monte_carlo_simulation(sample_df, "Monday", num_simulations=50)
    print(mc_result["average_summary"])
    print(mc_result["risk_summary"])