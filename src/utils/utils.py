"""
utils.py
    Shared utilities: structured logging, quarantine table, and small helpers
    used across dataset.py, features.py, and the modelling modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: 
    from src.config import LOG_FILE as _LOG_FILE_T 


cleaning_log: list[dict] = []
quarantine_records: list[pd.DataFrame] = []


def setup_logging(log_file: Path | str | None = None) -> logging.Logger:
    """
    Configure the root logger to write to *log_file* (append mode) and to
    stdout simultaneously.  Safe to call multiple times – duplicate handlers
    are not added.

    args:
        log_file:
            Path to the log file.  Falls back to the value in config if omitted.

    Returns
        logging.Logger
            The configured logger (name ``"collision_severity_predictor"``).
    """
    if log_file is None:
        from src.config import LOG_FILE
        log_file = LOG_FILE

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("collision_severity_predictor")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def log_action(
    step: str,
    rule: str,
    records_affected: int,
    action: str,
    rationale: str,
) -> None:
    """
    Record a data-cleaning decision to both the in-memory audit list and the
    rotating log file.

    args:
        step:              Pipeline stage (e.g. "Accuracy", "Completeness").
        rule:              Description of the rule or field involved.
        records_affected:  Number of rows (or columns) touched.
        action:            What was done (e.g. "Row deletion", "Median imputation").
        rationale:         Why this action was taken.
    """
    entry = {
        "step": step,
        "rule": rule,
        "records_affected": records_affected,
        "action": action,
        "rationale": rationale,
    }
    cleaning_log.append(entry)

    logger = logging.getLogger("collision_severity_predictor")
    logger.info(
        "%s | %s | %d records | %s | %s",
        step, rule, records_affected, action, rationale,
    )


def quarantine(df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    """
    Move rows matching *mask* into the global quarantine table and return the
    remaining rows.

    args:
        df:     DataFrame to filter.
        mask:   Boolean Series – ``True`` marks rows to quarantine.
        reason: Human-readable rejection reason stored alongside the rows.

    Returns:
        pd.DataFrame
            The input DataFrame with quarantined rows removed (a copy).
    """
    rejected = df[mask].copy()
    rejected["rejection_reason"] = reason
    quarantine_records.append(rejected)

    log_action(
        step="Accuracy",
        rule=reason,
        records_affected=int(mask.sum()),
        action="Reject → Quarantine",
        rationale="Critical field violation",
    )
    return df[~mask].copy()


def get_cleaning_report() -> pd.DataFrame:
    """
    Return the accumulated cleaning audit log as a tidy DataFrame.
    Useful for saving to ``reports/`` or printing in the final report.
    """
    if not cleaning_log:
        return pd.DataFrame(
            columns=["step", "rule", "records_affected", "action", "rationale"]
        )
    return pd.DataFrame(cleaning_log)


def get_quarantine_df() -> pd.DataFrame:
    """
    Concatenate all quarantined records into a single DataFrame.
    Returns an empty DataFrame if nothing was quarantined.
    """
    if not quarantine_records:
        return pd.DataFrame()
    return pd.concat(quarantine_records, ignore_index=True)