import json
import logging

import pandas as pd

import src.config as cfg
from src.data.load_data import (
    load_external_data,
    load_raw_data,
)
from src.utils import (
    log_action,
    save_stage_report,
)

logger = logging.getLogger("collision_severity_predictor")


def _prepare_collision_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["datetime_local"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    df["datetime_local"] = df["datetime_local"].dt.tz_localize(
        "Europe/London",
        ambiguous="NaT",
        nonexistent="shift_forward",
    )

    df["datetime_utc"] = df["datetime_local"].dt.tz_convert("UTC")

    df["_lat_r"] = df["latitude"].round(1)
    df["_lon_r"] = df["longitude"].round(1)

    df["datetime_utc_merge"] = df["datetime_utc"].dt.floor("h").dt.tz_localize(None)

    log_action(
        step="preparation for the merge",
        stage="merging",
        rule="date+time parsing and timezone conversion for collision data",
        records_affected=len(df),
        action="prepare for merge",
    )
    return df


def _prepare_weather_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize(None)
    log_action(
        step="preparation for the merge",
        stage="merging",
        rule="date+time parsing and timezone conversion for weather data",
        records_affected=len(df),
        action="prepare for merge",
    )
    return df


def merge_datasets(
    collision_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = collision_df.merge(
        weather_df,
        left_on=["datetime_utc_merge", "_lat_r", "_lon_r"],
        right_on=["datetime_utc", "_lat_r", "_lon_r"],
        how="left",
    )
    log_action(
        step="merging",
        stage="merging",
        rule="merging the collision and weather data",
        records_affected=len(merged),
        action="merge",
    )
    return merged


def validate_merge(merged_df: pd.DataFrame, weather_df: pd.DataFrame) -> dict:
    weather_cols = [c for c in weather_df.columns if c not in ("datetime_utc", "_lat_r", "_lon_r")]
    sample_col = weather_cols[0] if weather_cols else None

    missing_weather = merged_df[sample_col].isna().sum() if sample_col else 0
    merge_rate = (1 - missing_weather / len(merged_df)) * 100 if sample_col else 0.0

    report = {
        "merged_rows": len(merged_df),
        "merged_columns": len(merged_df.columns),
        "missing_weather_rows": missing_weather,
        "weather_merge_rate_pct": round(merge_rate, 2),
        "sample_weather_column_checked": sample_col,
    }
    with open(cfg.REPORTS_DIR / cfg.MERGING_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log_action(
        step="validation",
        stage="merging",
        rule="validating the merge",
        records_affected=missing_weather,
        action="validate",
    )
    return report


def save_merged_data(
    df: pd.DataFrame,
) -> None:
    out_path = cfg.INTERIM_DATA_DIR / cfg.INTERIM_OUTPUT_FILE
    df.to_csv(out_path, index=False)
    log_action(
        step="saving",
        stage="merging",
        rule="saving the merged data",
        records_affected=len(df),
        action="save",
    )


def merge_collision_weather() -> None:
    # setup_logging()

    collision_df = load_raw_data()
    weather_df = load_external_data()

    collision_df = _prepare_collision_datetime(collision_df)
    weather_df = _prepare_weather_datetime(weather_df)

    merged_df = merge_datasets(collision_df, weather_df)

    validate_merge(merged_df, weather_df)

    save_merged_data(merged_df)

    save_stage_report("merging")


if __name__ == "__main__":
    merge_collision_weather()
