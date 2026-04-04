from pathlib import Path
from typing import Dict, List

import importlib.util
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FILE_MAP = {
    "opd_patient_data": "opd_patient_data.csv",
    "department_wise_opd_data": "department_wise_opd_data.csv",
    "doctor_schedule_data": "doctor_schedule_data.csv",
    "appointment_data": "appointment_data.csv",
    "patient_category_data": "patient_category_data.csv",
}


def get_data_dir() -> Path:
    return DATA_DIR


def get_file_path(file_name: str) -> Path:
    return DATA_DIR / file_name


def get_required_file_paths() -> Dict[str, Path]:
    return {key: get_file_path(file_name) for key, file_name in FILE_MAP.items()}


def get_missing_files() -> List[str]:
    missing_files = []

    for _, file_name in FILE_MAP.items():
        file_path = get_file_path(file_name)
        if not file_path.exists():
            missing_files.append(file_name)

    return missing_files


def datasets_exist() -> bool:
    return len(get_missing_files()) == 0


def _load_sample_generator_module():
    generator_path = DATA_DIR / "sample_generator.py"

    if not generator_path.exists():
        raise FileNotFoundError(
            f"sample_generator.py not found at: {generator_path}"
        )

    spec = importlib.util.spec_from_file_location("sample_generator", generator_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load sample_generator module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_datasets_if_missing(
    start_date: str = "2026-01-01",
    num_days: int = 60,
    seed: int = 42,
) -> None:
    if datasets_exist():
        return

    module = _load_sample_generator_module()

    if not hasattr(module, "generate_all_datasets"):
        raise AttributeError(
            "generate_all_datasets() not found in data/sample_generator.py"
        )

    module.generate_all_datasets(
        start_date=start_date,
        num_days=num_days,
        seed=seed,
    )


def ensure_datasets_available(
    start_date: str = "2026-01-01",
    num_days: int = 60,
    seed: int = 42,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate_datasets_if_missing(
        start_date=start_date,
        num_days=num_days,
        seed=seed,
    )


def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")

    return df


def _postprocess_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = df.copy()

    if "date" in processed_df.columns:
        processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce")

    return processed_df


def load_opd_patient_data(auto_generate: bool = True) -> pd.DataFrame:
    if auto_generate:
        ensure_datasets_available()

    file_path = get_file_path(FILE_MAP["opd_patient_data"])
    df = load_csv(file_path)
    return _postprocess_common_columns(df)


def load_department_wise_opd_data(auto_generate: bool = True) -> pd.DataFrame:
    if auto_generate:
        ensure_datasets_available()

    file_path = get_file_path(FILE_MAP["department_wise_opd_data"])
    df = load_csv(file_path)
    return _postprocess_common_columns(df)


def load_doctor_schedule_data(auto_generate: bool = True) -> pd.DataFrame:
    if auto_generate:
        ensure_datasets_available()

    file_path = get_file_path(FILE_MAP["doctor_schedule_data"])
    df = load_csv(file_path)
    return _postprocess_common_columns(df)


def load_appointment_data(auto_generate: bool = True) -> pd.DataFrame:
    if auto_generate:
        ensure_datasets_available()

    file_path = get_file_path(FILE_MAP["appointment_data"])
    df = load_csv(file_path)
    return _postprocess_common_columns(df)


def load_patient_category_data(auto_generate: bool = True) -> pd.DataFrame:
    if auto_generate:
        ensure_datasets_available()

    file_path = get_file_path(FILE_MAP["patient_category_data"])
    df = load_csv(file_path)
    return _postprocess_common_columns(df)


def load_all_datasets(auto_generate: bool = True) -> Dict[str, pd.DataFrame]:
    if auto_generate:
        ensure_datasets_available()

    return {
        "opd_patient_data": load_opd_patient_data(auto_generate=False),
        "department_wise_opd_data": load_department_wise_opd_data(auto_generate=False),
        "doctor_schedule_data": load_doctor_schedule_data(auto_generate=False),
        "appointment_data": load_appointment_data(auto_generate=False),
        "patient_category_data": load_patient_category_data(auto_generate=False),
    }


def get_dataset_preview(df: pd.DataFrame, num_rows: int = 5) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.head(num_rows).copy()


def get_basic_dataset_info(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
    }


if __name__ == "__main__":
    ensure_datasets_available()
    all_data = load_all_datasets(auto_generate=False)

    print("All datasets loaded successfully.")
    for dataset_name, dataset_df in all_data.items():
        print(f"{dataset_name}: {dataset_df.shape}")