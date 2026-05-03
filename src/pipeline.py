from src.config import INTERIM_DATA_DIR
from src.data.cleaning import build_clean_dataset
from src.data.merging import merge_collision_weather
from src.features.build_features import main_features
from src.modeling.train import train_all_models
from src.modeling.eval import eval
from src.modeling.class_balancing import balance
import pandas as pd
from src.utils import setup_logging


def run_full_pipeline():
    setup_logging()
    merge_collision_weather()
    build_clean_dataset()
    main_features()
    balance()
    train_all_models()
    eval()

if __name__ == "__main__":
    run_full_pipeline()
