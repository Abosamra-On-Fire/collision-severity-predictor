from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

import src.config as cfg


stage_logs: dict[str, list[dict]] = {}
quarantine_records: list[pd.DataFrame] = []



def _stage_log_path(stage: str) -> Path:
    return cfg.LOG_DIR / f"{stage}.log"


def clear_all_logs() -> None:
    stage_logs.clear()
    quarantine_records.clear()
    log_dir = cfg.LOG_DIR
    for f in log_dir.glob("*.log"):
        f.unlink()


def setup_logging(log_file: Path | str | None = None) -> logging.Logger:
    if log_file is None:
        log_file = cfg.LOG_DIR / "pipeline.log"

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if log_file.exists():
        log_file.unlink()

    clear_all_logs()

    logger = logging.getLogger("collision_severity_predictor")

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def log_event(step: str, stage: str | None = None, **kwargs) -> None:
    stage = stage or step
    entry = {"step": step, **kwargs}
    
    if stage not in stage_logs:
        stage_logs[stage] = []
    stage_logs[stage].append(entry)

    logger = logging.getLogger("collision_severity_predictor")
    msg = " | ".join(f"{k}={v}" for k, v in entry.items())
    logger.info(msg)
    
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_stage_log_path(stage), "a", encoding="utf-8") as f:
        f.write(f"{ts} | INFO | {msg}\n")


def log_action(
    step: str,
    rule: str | None = None,
    records_affected: int | None = None,
    action: str | None = None,
    rationale: str | None = None,
    stage: str | None = None,
    **kwargs,
) -> None:
    log_event(
        step=step,
        stage=stage,
        rule=rule,
        records_affected=records_affected,
        action=action,
        rationale=rationale,
        **kwargs,
    )


def quarantine(df: pd.DataFrame, mask: pd.Series, reason: str, stage: str | None = None) -> pd.DataFrame:
    rejected = df[mask].copy()
    rejected["rejection_reason"] = reason
    quarantine_records.append(rejected)

    log_action(
        step="Accuracy",
        stage=stage,
        rule=reason,
        records_affected=int(mask.sum()),
        action="Reject → Quarantine",
        rationale="Critical field violation",
    )
    return df[~mask].copy()


def get_step_report(stage: str) -> pd.DataFrame:
    entries = stage_logs.get(stage, [])
    if not entries:
        return pd.DataFrame()
    return pd.DataFrame(entries)


def get_available_stages() -> list[str]:
    return list(stage_logs.keys())


def save_stage_report(stage: str, output_path: Path | str | None = None) -> Path:
    df = get_step_report(stage)
    if output_path is None:
        from src.config import REPORTS_DIR
        output_path = Path(REPORTS_DIR) / f"{stage}_report.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def get_quarantine_df() -> pd.DataFrame:
    if not quarantine_records:
        return pd.DataFrame()
    return pd.concat(quarantine_records, ignore_index=True)