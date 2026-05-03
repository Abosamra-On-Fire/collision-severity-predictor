from src.data.load_data import load_csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import numpy as np

from src import config as cfg
from src.utils import (
    log_action,
    quarantine,
)

import logging
import warnings
import os

logger = logging.getLogger("collision_severity_predictor")


def warning_to_log(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{category.__name__}: {message}")

warnings.showwarning = warning_to_log



####################### Feature Interactions ####################
def feature_interactions(
        df: pd.DataFrame,
        numerical_cols:list,
        categorical_cols:list
) -> pd.DataFrame:
    try:
        new_df = df.copy()
        if len(new_df) <=0 :
            return
        new_numerical_cols = numerical_cols.copy()
        new_categorical_cols = categorical_cols.copy()
        #? Arithmetic Complications

        new_df['casualties_per_vehicle'] = new_df['number_of_casualties'] / (new_df['number_of_vehicles'] + 1) 
        new_df['rain_intensity'] = new_df['wx_rain'] * new_df['wx_precipitation']
        new_df['wind_force'] = new_df['wx_wind_speed_10m'] * new_df['wx_wind_gusts_10m']

        new_df['visibility_risk'] = (
        (new_df['wx_cloud_cover'] / 100) * 0.4 +
        (new_df['wx_precipitation'] > 0).astype(int) * 0.4 +
        (new_df['weather_conditions'] == 7).astype(int) * 0.2 
    )
        new_numerical_cols = new_numerical_cols +[
            'casualties_per_vehicle',
            'rain_intensity',
            'wind_force',
            'visibility_risk'
        ]

        #? Boolean and Logical Combinations 

        new_df['is_bad_weather'] = (
        (new_df['wx_rain'] > 0) |
        (new_df['wx_wind_speed_10m'] > 20) |
        (new_df['wx_wind_gusts_10m'] > 20)
    ).astype(int)
        new_categorical_cols=new_categorical_cols+["is_bad_weather"]
        return new_df , new_numerical_cols ,  new_categorical_cols
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}")


####################### Feature Scaling ####################

def feature_scaling_fit(
        df: pd.DataFrame,
        numerical_cols: list
) -> ColumnTransformer:
    robust_col = cfg.ROBUST_COLS
    std_col = list(set(numerical_cols) - set(robust_col))
    preprocessor = ColumnTransformer(
    transformers=[
        ('rob', RobustScaler(), robust_col),
        ('std', StandardScaler(), std_col),
        
    ],
    remainder='passthrough'
)
    preprocessor.set_output(transform="pandas")
    preprocessor.fit(df)
    return preprocessor


def feature_scaling_transform(
        df: pd.DataFrame,
        preprocessor : ColumnTransformer
) -> pd.DataFrame:
    
    new_df = df.copy()
    new_df  = preprocessor.transform(new_df)
    new_df.columns = [
    col.split('__')[1] if '__' in col else col 
    for col in new_df.columns
]
    return new_df


####################### Feature Encoding ####################

def feature_encoding_fit(
        df: pd.DataFrame,
        categorical_cols: list
) -> BinaryEncoder:
    preprocessor = BinaryEncoder(cols=categorical_cols)
    preprocessor.set_output(transform="pandas")
    preprocessor.fit(df)
    return preprocessor


def feature_encoding_transform(
        df: pd.DataFrame,
        preprocessor: BinaryEncoder
)->pd.DataFrame: 
    new_df=df.copy()
    new_df = preprocessor.transform(new_df)
    new_df.columns = [
    col.split('__')[1] if '__' in col else col 
    for col in new_df.columns
]
    return new_df


####################### Feature Selection by variance thresholding ####################

def variance_thresholding_fit(
        df: pd.DataFrame,
        min_variance : float,
        numerical_cols:list
) -> VarianceThreshold:
    
    selector = VarianceThreshold(threshold=min_variance)
    df_numerical = df[numerical_cols]
    selector.fit(df_numerical)
    return selector


def variance_thresholding_transform(
        df: pd.DataFrame,
        selector : VarianceThreshold,
        numerical_cols : list
) -> tuple:
    
    new_df = df.copy()
    df_numerical = new_df[numerical_cols]
    df_numerical_selected = selector.transform(df_numerical)
    mask = selector.get_support()
    removed_cols = df_numerical.columns[~mask]
    kept_numerical_cols = df_numerical.columns[mask].tolist()
    new_df = new_df.drop(columns=removed_cols)
    return new_df, df_numerical_selected, kept_numerical_cols


####################### Feature Selection by correlation ####################

def correlation_based_selection (
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float,
        numerical_cols: list
) -> tuple:
    X_num = X[numerical_cols].copy()
    corr_matrix = X_num.corr(method='spearman').abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    target_corr = X_num.apply(lambda col: col.corr(y, method='spearman')).abs()
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            corr_val = upper.loc[row, col]
            if pd.notna(corr_val) and corr_val > threshold:
                if row in to_drop or col in to_drop:
                    continue
                if target_corr[row] < target_corr[col]:
                    to_drop.add(row)
                else:
                    to_drop.add(col)
    X_reduced = X.drop(columns=list(to_drop))

    kept_cols = [col for col in numerical_cols if col not in to_drop]
    return X_reduced, kept_cols, list(to_drop)


####################### Summary ####################


def summarize(
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        save_path: str
):

    summary = []
    summary.append(["rows", df_before.shape[0], df_after.shape[0]])
    summary.append(["columns", df_before.shape[1], df_after.shape[1]])

    





    num_before = df_before.select_dtypes(include=np.number)
    num_after = df_after.select_dtypes(include=np.number)

    summary.append([
        "mean_of_means",
        num_before.mean().mean(),
        num_after.mean().mean()
    ])

    summary.append([
        "mean_of_stds",
        num_before.std().mean(),
        num_after.std().mean()
    ])


    summary_df = pd.DataFrame(summary, columns=["metric", "before", "after"])

    summary_df.to_csv(save_path, index=False)


####################### Main Pipeline ####################

def main_features():
    target_col = cfg.TARGET_COL
    numerical_cols = cfg.NUMERICAL_COLS
    categorical_cols = cfg.CATEGORICAL_COLS
    df_train=load_csv(rf"{cfg.INTERIM_DATA_DIR}\{cfg.CLEANED_TRAIN_OUTPUT_FILE}")
    df_test=load_csv(rf"{cfg.INTERIM_DATA_DIR}\{cfg.CLEANED_TEST_OUTPUT_FILE}")


    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_val = df_test.drop(columns=[target_col])
    y_val = df_test[target_col]
    
    log_action(
        step="Feature interactions",
        stage="feature_engineering",
        rule="",
        records_affected=len(X_train)+len(X_val),
        action="Adding columns",
        rationale="To improve model expressiveness by exposing hidden nonlinear relationships and combined risk factors not captured by raw features",
    )
    X_train_with_interactions , numerical_cols_interaction, categorical_cols_interaction = feature_interactions(X_train,numerical_cols,categorical_cols)
    X_val_with_interactions , _, _  = feature_interactions(X_val,numerical_cols,categorical_cols)

    log_action(
        step="Feature Scaling",
        stage="feature_engineering",
        rule = "",
        records_affected=len(X_train_with_interactions)+len(X_val_with_interactions),
        action="Scaling Features by robust scaling",
        rationale="Data is mostly skewed so robust scaling would be the most accurate choice"
    )
    scaling_preprocessor = feature_scaling_fit (X_train_with_interactions,numerical_cols_interaction)

    X_train_scaled = feature_scaling_transform (X_train_with_interactions,scaling_preprocessor)
    X_val_scaled = feature_scaling_transform (X_val_with_interactions,scaling_preprocessor)

    log_action(
        step="Feature Encoding",
        stage="feature_engineering",
        rule = "",
        records_affected=len(X_train_scaled)+len(X_val_scaled),
        action="Encoding Features by binary scaling",
        rationale="Data is mostly nominal with high cardinality, so binary coding is the best choice"
    )
    encoding_preprocessor = feature_encoding_fit(X_train_scaled,categorical_cols_interaction)

    X_train_scaled_encoded = feature_encoding_transform(X_train_scaled,encoding_preprocessor)
    X_val_scaled_encoded = feature_encoding_transform(X_val_scaled,encoding_preprocessor)

    log_action(
        step="Variance Thresholding",
        stage="feature_engineering",
        rule = "",
        records_affected=len(X_train_scaled_encoded)+len(X_val_scaled_encoded),
        action="Columns Deletion",
        rationale="Any features having variance less than a certain threshold should be deleted"
    )
    selector = variance_thresholding_fit(X_train_scaled_encoded,cfg.VARIANCE_THRESHOLD,numerical_cols_interaction)

    X_train_after_variance_thresholding, X_train_selected_numerical, numerical_cols_thresholding = variance_thresholding_transform(X_train_scaled_encoded,selector,numerical_cols_interaction)
    X_val_after_variance_thresholding, X_val_selected_numerical, _ = variance_thresholding_transform (X_val_scaled_encoded,selector,numerical_cols_interaction)

    log_action(
        step="Correlation Based Selection",
        stage="feature_engineering",
        rule = "",
        records_affected=len(X_train_after_variance_thresholding)+len(X_val_after_variance_thresholding),
        action="Columns Deletion",
        rationale="For each correlated pair of columns, we drop the least correlated with the target"
    )
    X_train_corr, numerical_cols_correlation, dropped_cols =correlation_based_selection(X_train_after_variance_thresholding,y_train,cfg.CORRELATION_THRESHOLD,numerical_cols_thresholding)
    X_val_corr = X_val_after_variance_thresholding.drop(columns= dropped_cols)
    
    train_final = X_train_corr.copy()
    val_final = X_val_corr.copy()

    train_final[target_col] = y_train.values
    val_final[target_col] = y_val.values
    
    train_path = os.path.join(cfg.PROCESSED_DATA_DIR, cfg.TRAIN_OUTPUT_FILE)
    val_path = os.path.join(cfg.PROCESSED_DATA_DIR, cfg.TEST_OUTPUT_FILE)

    train_final.to_csv(train_path, index=False)
    val_final.to_csv(val_path, index=False)

    summarize(X_train,X_train_corr,rf"{cfg.REPORTS_DIR}/features_summary.csv")
    print(train_final.columns)


###########################################
if __name__ == "__main__":
    main_features()