from src.data.load_data import load_csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import numpy as np


target_col = 'collision_severity'

base_numerical_cols = [
    'longitude',
    'latitude',
    'number_of_vehicles',
    'number_of_casualties',
    'wx_temperature_2m',
    'wx_relative_humidity_2m',
    'wx_precipitation',
    'wx_rain',
    'wx_snowfall',
    'wx_snow_depth',
    'wx_wind_speed_10m',
    'wx_wind_direction_10m',
    'wx_wind_gusts_10m',
    'wx_visibility',
    'wx_surface_pressure',
    'wx_cloud_cover',
]

categorical_cols = [
    'day_of_week',
    'speed_limit',
    'time',
    'first_road_class',
    'road_type',
    'junction_detail',
    'junction_control',
    'second_road_class',
    'pedestrian_crossing',
    'light_conditions',
    'weather_conditions',
    'road_surface_conditions',
    'special_conditions_at_site',
    'carriageway_hazards',
    'urban_or_rural_area',
    'did_police_officer_attend_scene_of_accident',
    'wx_weather_code',
    'wx_is_day']


engineered_numerical_cols = [
    'casualties_per_vehicle',
    'rain_intensity',
    'wind_force',
    'visibility_risk'
]


numerical_cols = base_numerical_cols + engineered_numerical_cols

####################### Feature Interactions ####################
def feature_interactions(
        df: pd.DataFrame
) -> pd.DataFrame:
    
    new_df = df.copy()

    #? Arithmetic Complications

    new_df['casualties_per_vehicle'] = new_df['number_of_casualties'] / (new_df['number_of_vehicles'] + 1) 
    new_df['rain_intensity'] = new_df['wx_rain'] * new_df['wx_precipitation']
    new_df['wind_force'] = new_df['wx_wind_speed_10m'] * new_df['wx_wind_gusts_10m']

    new_df['visibility_risk'] = (
    (new_df['wx_cloud_cover'] / 100) * 0.4 +
    (new_df['wx_precipitation'] > 0).astype(int) * 0.4 +
    (new_df['weather_conditions'] == 7).astype(int) * 0.2 
)
    
    # numerical_cols.extend([
    #     'casualties_per_vehicle',
    #     'rain_intensity',
    #     'wind_force',
    #     'visibility_risk'
    # ])

    #? Boolean and Logical Combinations 

    new_df['is_bad_weather'] = (
    (new_df['wx_rain'] > 0) |
    (new_df['wx_wind_speed_10m'] > 20) |
    (new_df['wx_wind_gusts_10m'] > 20)
).astype(int)
    
    return new_df

####################### Feature Scaling ####################

def feature_scaling_fit(
        df: pd.DataFrame
) -> ColumnTransformer:
    robust_col = [ 
    'number_of_vehicles',  
    'number_of_casualties', 
    'rain_intensity' 
]
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
        df: pd.DataFrame
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
        min_variance : float
) -> VarianceThreshold:
    
    selector = VarianceThreshold(threshold=min_variance)
    df_numerical = df[numerical_cols]
    selector.fit(df_numerical)
    return selector


def variance_thresholding_transform(
        df: pd.DataFrame,
        selector : VarianceThreshold
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
    print(to_drop)
    return X_reduced, kept_cols, list(to_drop)


def main():
    df=load_csv(r"E:\Data_Science_Project\collision-severity-predictor\data\raw\accidents_with_weather.csv")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Lazem asplit abl ay haga 3shan my7salsh leakage
    # Hsplit le training set we validation set 80-20
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
    
    X_train_with_interactions = feature_interactions(X_train)
    X_val_with_interactions   = feature_interactions(X_val)

    # Ht3lm el stats mnel training bs b3dha happly them 3la kolo
    scaling_preprocessor = feature_scaling_fit (X_train_with_interactions)

    X_train_scaled = feature_scaling_transform (X_train_with_interactions,scaling_preprocessor)
    X_val_scaled = feature_scaling_transform (X_val_with_interactions,scaling_preprocessor)

    # nfsel kalam brdo 3shan my7salsh leakage
    encoding_preprocessor = feature_encoding_fit(X_train_scaled)

    X_train_scaled_encoded = feature_encoding_transform(X_train_scaled,encoding_preprocessor)
    X_val_scaled_encoded = feature_encoding_transform(X_val_scaled,encoding_preprocessor)

    # nfsel kalam brdo 3shan my7salsh leakage
    selector = variance_thresholding_fit(X_train_scaled_encoded,0.01)

    X_train_after_variance_thresholding, X_train_selected_numerical, kept_numerical_cols = variance_thresholding_transform(X_train_scaled_encoded,selector)
    X_val_after_variance_thresholding, X_val_selected_numerical, _ = variance_thresholding_transform (X_val_scaled_encoded,selector)

    # dlw2ty hdrop bel correlation based
    X_train_corr, numerical_cols, dropped_cols =correlation_based_selection(X_train_after_variance_thresholding,y_train,0.9,kept_numerical_cols)
    X_val_corr = X_val_after_variance_thresholding.drop(columns= dropped_cols)
    
if __name__ == "__main__":
    main()



