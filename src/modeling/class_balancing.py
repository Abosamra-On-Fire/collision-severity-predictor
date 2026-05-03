from pathlib import Path

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import pandas as pd
from src.data.load_data import load_csv

from src import config as cfg


def undersample_data(df, target_col, method="random"):
    """
    Undersample the majority classes.

    Methods:
    - 'random': Randomly remove samples from majority classes
    - 'tomek': Remove Tomek Links (noisy boundary samples)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nBefore undersampling ({method}):")
    print(y.value_counts().sort_index())

    if method == "random":
        sampler = RandomUnderSampler(random_state=42)
    elif method == "tomek":
        sampler = TomekLinks()
    else:
        raise ValueError(f"Unknown method: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    print(f"\nAfter undersampling ({method}):")
    print(pd.Series(y_resampled).value_counts().sort_index())

    
    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_col] = y_resampled

    return df_balanced


def balance():

    train_df = load_csv(cfg.PROCESSED_DATA_DIR / cfg.PRO_TRAIN_OUTPUT_FILE)
    train_balanced = undersample_data(train_df, cfg.TARGET_COL, method="tomek")
    train_balanced.to_csv(cfg.PROCESSED_DATA_DIR / cfg.BALANCED_TRAIN_OUTPUT_FILE, index=False)
if __name__ == "__balance__":
    balance()
