import logging
from pathlib import Path
from typing import Union

import pandas as pd

from src import config as cfg

logger = logging.getLogger("collision_severity_predictor")


def load_csv(filepath: Union[str, Path], encoding: str = "utf-8", **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        filepath: Path to the CSV file
        encoding: File encoding (default: 'utf-8')
        **kwargs: Additional arguments to pass to pd.read_csv()

    Returns:
        DataFrame containing the loaded data

    Example:
        df = load_csv('data/raw/dataset.csv')
    """
    try:
        filepath = Path(filepath)
        # log_action(step="loading", rule="loading raw data", action="load")
        df = pd.read_csv(filepath, encoding=encoding, **kwargs)
        # log_action(step="loading", rule=f"Successfully loaded {len(df)} rows and {len(df.columns)} columns", action="load")

        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw collision dataset from data/raw/.
    Returns
        pd.DataFrame
            Raw collision data.
    """
    path = cfg.RAW_DATA_DIR / cfg.RAW_COLLISION_FILE
    df = load_csv(path)
    return df


def load_external_data() -> pd.DataFrame:
    """
    Load the external weather dataset from data/external/.
    Returns
        pd.DataFrame
            External weather data.
    """
    path = cfg.EXTERNAL_DATA_DIR / cfg.EXTERNAL_WEATHER_FILE
    df = load_csv(path)
    return df
