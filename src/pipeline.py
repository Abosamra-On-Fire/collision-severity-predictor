from src.config import INTERIM_DATA_DIR
from src.data.cleaning import build_clean_dataset
from src.data.merging import merge_collision_weather
from src.features.build_features import main_features
from src.modeling.train import train_all_models
from src.modeling.eval import eval
import pandas as pd
from src.utils import setup_logging


def run_full_pipeline():
    setup_logging()
    print("=" * 50)
    print("STEP 1: Data Acquisition")
    print("=" * 50)
    merge_collision_weather()

    print("=" * 50)
    print("STEP 2: Cleaning")
    print("=" * 50)
    # df = pd.read_csv(DATA_DIR / "interim" / "merged.csv")
    df_train, df_test = build_clean_dataset(INTERIM_DATA_DIR / "accidents_with_weather.csv")
    print(df_train.isnull().sum())
    print(df_test.isnull().sum())
    print(len(df_train))
    print(len(df_test))

    # print("=" * 50)
    # print("STEP 3: Feature Engineering")
    # print("=" * 50)
    # df = engineer_features(df)

    # print("=" * 50)
    # print("STEP 4: Train/Test Split")
    # print("=" * 50)
    # X = df.drop(columns=["collision_severity", "datetime"])
    # y = df["collision_severity"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # processed_dir = DATA_DIR / "processed"
    # processed_dir.mkdir(parents=True, exist_ok=True)
    # X_train.to_csv(processed_dir / "X_train.csv", index=False)
    # X_test.to_csv(processed_dir / "X_test.csv", index=False)
    # y_train.to_csv(processed_dir / "y_train.csv", index=False)
    # y_test.to_csv(processed_dir / "y_test.csv", index=False)

    # print("=" * 50)
    # print("STEP 5: Model Training")
    # print("=" * 50)
    # train_all_models()

    # print("=" * 50)
    # print("PIPELINE COMPLETE")
    # print("=" * 50)
    main_features()
    train_all_models()
    eval()

if __name__ == "__main__":
    run_full_pipeline()
