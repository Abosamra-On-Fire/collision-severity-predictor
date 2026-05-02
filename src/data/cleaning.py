from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.load_data import load_csv

from src import config as cfg
from src.utils import (
    log_action,
    quarantine,
    setup_logging,
)

logger = logging.getLogger("collision_severity_predictor")

def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop administrative, redundant, and leakage columns documented in the
    Phase 1 report.  Unknown column names are silently ignored.
    Returns:
        pd.DataFrame  (copy)
    """
    present = [c for c in cfg.COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=present)
    log_action(
        step="Column Pruning",
        rule="Admin / leakage columns",
        records_affected=len(present),
        action="Column deletion",
        rationale="Documented in Phase 1 report – not predictive or leakage risk",
    )
    logger.info("After column pruning: %s", df.shape)
    return df.copy()

def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast categorical columns to the category dtype and time to
    datetime.time objects.

    Returns
        pd.DataFrame  (modified in-place copy)
    """
    df = df.copy()

    for col in cfg.CATEGORICAL_COLS:
        if col in df.columns and df[col].dtype in ("int64", "int32", "float64"):
            df[col] = df[col].astype("category")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"], format="%H:%M", errors="coerce"
        ).dt.time
        log_action(
            step="Consistency",
            rule="time format",
            records_affected=len(df),
            action="Coerce to datetime.time",
            rationale="Standardise temporal format",
        )

    return df


def validate_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reject rows with physically impossible field values by moving them to the
    quarantine table (see utils.quarantine).

    Checks
        * Latitude / longitude outside UK bounding box.
        * number_of_vehicles or number_of_casualties ≤ 0.
    Returns
        pd.DataFrame  (rows removed)
    """
    geo_mask = (
        (df["latitude"]  < cfg.UK_LAT_MIN) | (df["latitude"]  > cfg.UK_LAT_MAX) |
        (df["longitude"] < cfg.UK_LON_MIN) | (df["longitude"] > cfg.UK_LON_MAX)
    )
    df = quarantine(df, geo_mask, "latitude/longitude outside UK bounds")

    count_mask = (
        (df["number_of_vehicles"]  <= 0) |
        (df["number_of_casualties"] <= 0)
    )
    df = quarantine(df, count_mask, "number_of_vehicles or number_of_casualties <= 0")

    logger.info("After accuracy quarantine: %s", df.shape)
    return df


def fix_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sentinel values (e.g. -1 used as "unknown") with NaN so that
    downstream imputation logic handles them uniformly.

    Returns
        pd.DataFrame  (copy)
    """
    df = df.copy()

    for col, codes in cfg.CODED_MISSING.items():
        if col not in df.columns:
            continue
        count = int(df[col].isin(codes).sum())
        if count:
            df[col] = df[col].replace(codes, np.nan)
            log_action(
                step="Consistency",
                rule=f"{col} coded as {codes}",
                records_affected=count,
                action="Replace with NaN",
                rationale="Standardise missing value representation",
            )
    return df

def handle_completeness(
    df: pd.DataFrame,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the completeness strategy.

    * Fit mode    params is None.  Computes and returns params.
    * Transform   pass the dict returned from the train call.  Applies the
      same columns-drops / imputation values without looking at test stats.
    
    args:
        df: DataFrame
        params: dict

    Returns
        (df_clean, params)
    """
    df = df.copy()
    fit = params is None         

    if fit:
        params = {
            "cols_to_drop": [],
            "mcar_cols": [],
            "numerical_impute": {},
            "categorical_impute": {},
        }

    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="category").columns.tolist()
    all_features = set(numerical_cols + categorical_cols)


    if fit:
        missing_rates = df.isnull().mean()
        params["cols_to_drop"] = [
            c for c in missing_rates.index
            if c in all_features and missing_rates[c] >= cfg.COLS_NULL_PREC
        ]
        for col in params["cols_to_drop"]:
            log_action(
                step="Completeness",
                rule=f"{col} {missing_rates[col]:.0%} missing",
                records_affected=len(df),
                action="Column deletion",
                rationale=f"{missing_rates[col]:.0%} missing – cannot impute reliably",
            )

    for col in params["cols_to_drop"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            if col in numerical_cols:
                numerical_cols.remove(col)
            if col in categorical_cols:
                categorical_cols.remove(col)


    if fit:
        mcar_present = [c for c in cfg.MCAR_COLS if c in df.columns]
        params["mcar_cols"] = [
            c for c in mcar_present
            if 0 < df[c].isnull().mean() < cfg.ROW_NULL_PREC
        ]

    low_missing_mcar = [c for c in params["mcar_cols"] if c in df.columns]
    if low_missing_mcar:
        mcar_mask = df[low_missing_mcar].isnull().any(axis=1)
        mcar_count = int(mcar_mask.sum())
        if mcar_count:
            df = df.loc[~mcar_mask].copy()
            if fit:
                log_action(
                    step="Completeness",
                    rule=f"{', '.join(low_missing_mcar)} missing",
                    records_affected=mcar_count,
                    action="Row deletion",
                    rationale=f"<{cfg.ROW_NULL_PREC:.0%} missing, MCAR pattern",
                )

    for col in numerical_cols:
        if col not in df.columns:
            continue

        if fit:
            params["numerical_impute"][col] = df[col].median()

        n_miss = int(df[col].isnull().sum())
        if n_miss > 0:
            fill_val = params["numerical_impute"].get(col)
            if fill_val is not None:
                df[col] = df[col].fillna(fill_val)
                if fit:
                    log_action(
                        step="Completeness",
                        rule=f"{col} missing",
                        records_affected=n_miss,
                        action="Median imputation",
                        rationale="Skewed distribution – robust to outliers",
                    )

    for col in categorical_cols:
        if col not in df.columns:
            continue

        if fit:
            mode_val = df[col].mode()
            params["categorical_impute"][col] = (
                mode_val.iloc[0] if not mode_val.empty else None
            )

        n_miss = int(df[col].isnull().sum())
        if n_miss > 0:
            fill_val = params["categorical_impute"].get(col)
            if fill_val is not None:
                df[col] = df[col].fillna(fill_val)
                if fit:
                    log_action(
                        step="Completeness",
                        rule=f"{col} missing",
                        records_affected=n_miss,
                        action="Mode imputation",
                        rationale="Clear dominant category",
                    )

    if fit:
        logger.info("After completeness handling: %s", df.shape)

    return df, params

def _handle_mode_heavy(
    df: pd.DataFrame,
    col: str,
    dominant_val: float,
) -> pd.DataFrame:
    """Cap / remove outliers among non-dominant values when a column is
    dominated by a single value (mode > DOMINANT_THRESHOLD)."""
    non_dom_mask = df[col] != dominant_val
    non_dom_vals = df.loc[non_dom_mask, col]

    if len(non_dom_vals) < 10:
        log_action(
            step="Outliers",
            rule=f"{col} mode-heavy ({dominant_val})",
            records_affected=len(non_dom_vals),
            action="Skipped",
            rationale="Insufficient non-dominant samples",
        )
        return df

    lower = non_dom_vals.quantile(0.05)
    upper = non_dom_vals.quantile(0.95)
    outlier_mask = non_dom_vals[(non_dom_vals < lower) | (non_dom_vals > upper)].index
    full_mask = df.index.isin(outlier_mask)
    count = int(full_mask.sum())

    if count == 0:
        return df

    if count > cfg.OUTLIER_RATIO_THRESHOLD * len(df):
        clipped = df.loc[non_dom_mask, col].clip(lower=lower, upper=upper)
        if pd.api.types.is_integer_dtype(df[col]) and (clipped % 1).any():
            df[col] = df[col].astype(float)
        df.loc[non_dom_mask, col] = clipped
        log_action("Outliers", f"{col} mode-heavy IQR", count, "Capping",
                   f"Non-dominant values capped, dominant={dominant_val}")
    else:
        df = df[~full_mask].copy()
        log_action("Outliers", f"{col} mode-heavy IQR", count, "Removal",
                   f"Non-dominant outliers removed, dominant={dominant_val}")

    return df


def _outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Standard IQR cap/remove for columns without a dominant mode."""
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return df

    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outlier_mask = (df[col] < lower) | (df[col] > upper)
    count = int(outlier_mask.sum())
    if count == 0:
        return df

    if count > cfg.OUTLIER_RATIO_THRESHOLD * len(df):
        clipped = df[col].clip(lower=lower, upper=upper)
        if pd.api.types.is_integer_dtype(df[col]) and (clipped % 1).any():
            df[col] = df[col].astype(float)
        df[col] = clipped
        log_action("Outliers", f"{col} IQR", count, "Capping",
                   "Preserve dataset size, reduce outlier influence")
    else:
        df = df[~outlier_mask].copy()
        log_action("Outliers", f"{col} IQR", count, "Removal", "Confirmed errors")

    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and handle outliers in every numerical column (excluding lat/lon
    which were already validated for accuracy).

    Applies mode-heavy logic when a single value occupies > DOMINANT_THRESHOLD
    fraction of the column; falls back to IQR otherwise.

    Returns
        pd.DataFrame
    """
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    skip = {"longitude", "latitude"}

    for col in numerical_cols:
        if col not in df.columns or col in skip:
            continue
        valid = df[col].dropna()
        if valid.empty:
            continue

        dominant_val = valid.mode().iloc[0]
        dominant_ratio = (df[col] == dominant_val).mean()

        if dominant_ratio > cfg.DOMINANT_THRESHOLD:
            df = _handle_mode_heavy(df, col, dominant_val)
        else:
            df = _outliers_iqr(df, col)

    logger.info("After outlier handling: %s", df.shape)
    return df

def invert_severity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw dataset encodes severity as 1=Fatal, 3=Slight (inverse order).
    Map to 1=Slight, 2=Serious, 3=Fatal via  4 - original.

    Returns
        pd.DataFrame  (copy)
    """
    df = df.copy()
    df[cfg.TARGET_COL] = 4 - df[cfg.TARGET_COL]
    log_action(
        step="Consistency",
        rule="collision_severity label order",
        records_affected=len(df),
        action="Label inversion (4 - x)",
        rationale="Align ordinal encoding: 3 = most severe",
    )
    return df

def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80 / 20 train-test split.

    The test set is **not** imputed here; imputation is fitted on train and
    applied separately to avoid leakage.

    Returns
        (df_train, df_test)
    """
    df_train, df_test = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=df[cfg.TARGET_COL],
    )

    logger.info("Train shape: %s | Test shape: %s", df_train.shape, df_test.shape)
    return df_train, df_test




def build_clean_dataset(
    merged_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    End-to-end data pipeline: load → prune → cast → validate → clean →
    outliers → split.

    args:
        merged_path: Path to the raw CSV (defaults to config.RAW_DATA_FILE).

    Returns
        df_train, df_test, numerical_cols, categorical_cols
        The last two lists reflect any columns removed during cleaning.
    """
    setup_logging()

    df = load_csv(merged_path)
    df = drop_unwanted_columns(df)
    df = cast_column_types(df)
    df = validate_accuracy(df)
    df = fix_consistency(df)

    df_train, df_test = split_dataset(df)

    df_train, params = handle_completeness(df_train)
    df_test, _ = handle_completeness(df_test, params=params)

    df_train = handle_outliers(df_train) #momken na5aly el test t-get bounded be el values aly gabnaha men el train

    df_train = invert_severity_labels(df_train)
    df_test = invert_severity_labels(df_test)
 
    return df_train, df_test