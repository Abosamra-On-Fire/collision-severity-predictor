import pandas as pd
import logging
from pathlib import Path
from typing import  Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv(
    filepath: Union[str, Path],
    encoding: str = 'utf-8',
    **kwargs
) -> pd.DataFrame:
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
        logger.info(f"Loading CSV file: {filepath}")
        
        df = pd.read_csv(filepath, encoding=encoding, **kwargs)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        raise
