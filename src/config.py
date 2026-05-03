from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_COLLISION_FILE = "road-casualty-statistics-raw2025.csv"
EXTERNAL_WEATHER_FILE = "merged_weather.csv"

INTERIM_OUTPUT_FILE = "accidents_with_weather.csv"

CLEANED_TRAIN_OUTPUT_FILE = "train_clean.csv"
CLEANED_TEST_OUTPUT_FILE = "test_clean.csv"

REPORTS_DIR = PROJ_ROOT / "reports"

MERGING_REPORT_FILE = "merging_report.json"

LOG_DIR = REPORTS_DIR / "logs"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_OUTPUT_FILE = "train.csv"
VAL_OUTPUT_FILE = "val.csv"
TEST_OUTPUT_FILE = "test.csv"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


TARGET_COL = "collision_severity"

SEVERITY_LABELS = {1: "Slight", 2: "Serious", 3: "Fatal"}


COLUMNS_TO_DROP = [
    "collision_index",
    "collision_ref_no",
    "collision_year",
    "location_easting_osgr",
    "location_northing_osgr",
    "local_authority_district",
    "local_authority_ons_district",
    "local_authority_highway",
    "local_authority_highway_current",
    "lsoa_of_accident_location",
    "first_road_number",
    "second_road_number",
    "junction_detail_historic",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "carriageway_hazards_historic",
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
    "trunk_road_flag",
    "police_force",
    "datetime_local",
    "datetime_utc_x",
    "_lat_r",
    "_lon_r",
    "datetime_utc_merge",
    "datetime_utc_y",
    "accident_hour",
    "date",
]

NUMERICAL_COLS = [
    "longitude",
    "latitude",
    "number_of_vehicles",
    "number_of_casualties",
    "wx_temperature_2m",
    "wx_relative_humidity_2m",
    "wx_precipitation",
    "wx_rain",
    "wx_snowfall",
    "wx_snow_depth",
    "wx_wind_speed_10m",
    "wx_wind_direction_10m",
    "wx_wind_gusts_10m",
    "wx_surface_pressure",
    "wx_cloud_cover",
]

CATEGORICAL_COLS = [
    "day_of_week",
    "speed_limit",
    "time",
    "first_road_class",
    "road_type",
    "junction_detail",
    "second_road_class",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "carriageway_hazards",
    "urban_or_rural_area",
    "did_police_officer_attend_scene_of_accident",
    "wx_weather_code",
    "wx_is_day",
]


CODED_MISSING: dict[str, list] = {
    "speed_limit": [-1],
    "junction_detail": [-1],
    "second_road_class": [-1],
    "pedestrian_crossing": [-1],
    "special_conditions_at_site": [-1],
    "carriageway_hazards": [-1],
    "light_conditions": [-1],
    "weather_conditions": [-1],
    "road_surface_conditions": [-1],
    "did_police_officer_attend_scene_of_accident": [-1],
}


ROBUST_COLS = [
    "number_of_vehicles",
    "number_of_casualties",
    "rain_intensity",
]

# Outlier Handling
DOMINANT_THRESHOLD = 0.30
OUTLIER_RATIO_THRESHOLD = 0.001


# Geographic bounds for the UK
UK_LAT_MIN, UK_LAT_MAX = 49, 61
UK_LON_MIN, UK_LON_MAX = -11, 2

RANDOM_STATE = 42
TEST_SIZE = 0.20

MCAR_COLS = ["latitude", "longitude", "speed_limit"]


ROW_NULL_PREC = 0.05
COLS_NULL_PREC = 0.40

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.9


MODELS_DIR = PROJ_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


N_SPLITS = 5

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db" 
MLFLOW_EXPERIMENT_NAME = "collision_severity_classification"



SEVERITY_COST_MATRIX: dict[tuple[int, int], float] = {
    (0, 1): 3,   
    (0, 2): 10, 
    (1, 0): 1,  
    (1, 2): 5,  
    (2, 0): 1,  
    (2, 1): 1,  
}

CV_FOLDS   = 2
N_ITER_RS  = 3
SCORING_CV = "f1_weighted"


MLP_HIDDEN_DIM  = 256
MLP_NUM_CLASSES = 3
MLP_BATCH_SIZE  = 64
MLP_EPOCHS      = 5
MLP_LR          = 1e-3
MLP_PATIENCE    = 15
MLP_DROPOUT     = 0.4
MLP_WEIGHT_DECAY = 1e-4

# Figures
PIE_CHART_FOR_SEVERITY = "collision_severity_pie_chart.png"
GEO_DISTRIBUTION = "geo_distribution.png"
HOURLY_ACCIDENTS = "hourly.png"
RAIN_VS_SEVERITY = "rain_vs_severity.png"
AVG_COLLISION_VS_ROAD_TYPE = "avg_collision_vs_road_type.png"
# UNDERSAMPLE_STRATEGY = {1: 17_810, 2: 5_962, 3: 556}

# MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "collision_severity")
# MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

# RF_PARAMS = dict(
#     n_estimators=200,
#     max_depth=None,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features="sqrt",
#     class_weight="balanced",
#     random_state=RANDOM_STATE,
#     n_jobs=-1,
# )

# XGB_PARAMS = dict(
#     n_estimators=200,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     min_child_weight=1,
#     objective="multi:softprob",
#     num_class=3,
#     eval_metric="mlogloss",
#     random_state=RANDOM_STATE,
#     n_jobs=-1,
# )

# CATBOOST_PARAMS = dict(
#     iterations=200,
#     depth=6,
#     learning_rate=0.1,
#     l2_leaf_reg=3,
#     random_seed=RANDOM_STATE,
#     loss_function="MultiClass",
#     eval_metric="MultiClass",
#     verbose=False,
#     thread_count=-1,
# )

# LGBM_PARAMS = dict(
#     n_estimators=200,
#     max_depth=-1,
#     num_leaves=31,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     min_child_samples=5,
#     min_child_weight=0.001,
#     reg_alpha=0,
#     reg_lambda=0,
#     objective="multiclass",
#     num_class=3,
#     random_state=RANDOM_STATE,
#     n_jobs=-1,
#     verbosity=-1,
# )

# MLP_PARAMS = dict(
#     hidden_dim=256,
#     dropout=0.4,
#     batch_size=64,
#     epochs=50,
#     learning_rate=1e-3,
#     patience=15,
#     weight_decay=1e-4,
# )
