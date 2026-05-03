"""
Class Balancing Script
======================
Handles class imbalance using undersampling techniques and saves balanced data.
"""

from pathlib import Path

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as cfg

# =========================
# CONFIG
# =========================
TARGET_COL = "collision_severity"

INPUT_TRAIN = rf"{cfg.PROCESSED_DATA_DIR}\train.csv"
INPUT_VAL = rf"{cfg.PROCESSED_DATA_DIR}\val.csv"

OUTPUT_DIR = rf"{cfg.PROCESSED_DATA_DIR}\balanced"


# =========================
# LOAD DATA
# =========================
def load_data(train_path, val_path, target_col):
    """Load train and val data, split val into val and test."""
    print("Loading data...")

    train_df = pd.read_csv(train_path)
    val_test_df = pd.read_csv(val_path)

    # Split val into actual val and test
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.5, stratify=val_test_df[target_col], random_state=42
    )

    print(f"Train: {len(train_df):,} samples")
    print(f"Val  : {len(val_df):,} samples")
    print(f"Test : {len(test_df):,} samples")

    return train_df, val_df, test_df


# =========================
# UNDERSAMPLE
# =========================
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

    # Combine back into dataframe
    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target_col] = y_resampled

    return df_balanced


# =========================
# SAVE
# =========================
def save_splits(train_df, val_df, test_df, output_dir):
    """Save the balanced datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"\nSaved to {output_dir}:")
    print(f"  train.csv: {len(train_df):,} samples")
    print(f"  val.csv  : {len(val_df):,} samples")
    print(f"  test.csv : {len(test_df):,} samples")


# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("CLASS BALANCING - UNDERSAMPLING")
    print("=" * 60)

    # Load data
    train_df, val_df, test_df = load_data(INPUT_TRAIN, INPUT_VAL, TARGET_COL)

    # Clean training set with Tomek Links
    # This removes noisy boundary samples, NOT full balancing
    train_balanced = undersample_data(train_df, TARGET_COL, method="tomek")

    # Don't undersample val and test - keep them as-is for evaluation

    # Save all splits
    save_splits(train_balanced, val_df, test_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
