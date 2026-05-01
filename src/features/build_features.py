from src.data.load_data import load_csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
import pandas as pd



target_col = 'collision_severity'

numerical_cols = [
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
    
    #? Boolean and Logical Combinations 

    new_df['is_bad_weather'] = (
    (new_df['wx_rain'] > 0) |
    (new_df['wx_wind_speed_10m'] > 20) |
    (new_df['wx_wind_gusts_10m'] > 20)
).astype(int)
    
    return new_df



def feature_scaling(
        df: pd.DataFrame
) -> pd.DataFrame:
    pass


def main():
    df=load_csv(r"E:\Data_Science_Project\collision-severity-predictor\data\raw\accidents_with_weather.csv")
    df_after_interactions = feature_interactions(df)
    

if __name__ == "__main__":
    main()



