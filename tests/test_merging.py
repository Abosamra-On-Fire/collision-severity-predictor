from __future__ import annotations

import json
from datetime import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import src.data.merging as merging



@pytest.fixture
def collision_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["01/01/2023", "02/01/2023", "03/01/2023"],
            "time": [time(8, 30), time(14, 0), time(23, 45)],
            "latitude": [51.5074, 51.5074, 53.4808],
            "longitude": [-0.1278, -0.1278, -2.2426],
            "collision_id": [1, 2, 3],
        }
    )


@pytest.fixture
def weather_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime_utc": [
                "2023-01-01 08:00:00",
                "2023-01-01 09:00:00",
                "2023-01-02 14:00:00",
                "2023-01-03 23:00:00",
            ],
            "_lat_r": [51.5, 51.5, 51.5, 53.5],
            "_lon_r": [-0.1, -0.1, -0.1, -2.2],
            "temperature": [5.0, 6.0, 7.0, 2.0],
            "humidity": [80, 82, 78, 90],
        }
    )


@pytest.fixture
def prepared_collision(collision_df) -> pd.DataFrame:
    """Collision data after datetime preparation."""
    return merging._prepare_collision_datetime(collision_df)


@pytest.fixture
def prepared_weather(weather_df) -> pd.DataFrame:
    """Weather data after datetime preparation."""
    return merging._prepare_weather_datetime(weather_df)





class TestPrepareCollisionDatetime:
    def test_creates_expected_columns(self, collision_df):
        result = merging._prepare_collision_datetime(collision_df)
        assert "datetime_local" in result.columns
        assert "datetime_utc" in result.columns
        assert "_lat_r" in result.columns
        assert "_lon_r" in result.columns
        assert "datetime_utc_merge" in result.columns

    def test_datetime_parsing(self, collision_df):
        result = merging._prepare_collision_datetime(collision_df)
        
        expected = pd.Timestamp("2023-01-01 08:30:00", tz="UTC")
        assert result["datetime_utc"].iloc[0] == expected

    def test_rounding(self, collision_df):
        result = merging._prepare_collision_datetime(collision_df)
        assert result["_lat_r"].iloc[0] == 51.5
        assert result["_lon_r"].iloc[0] == -0.1

    def test_floor_to_hour(self, collision_df):
        result = merging._prepare_collision_datetime(collision_df)
        
        assert result["datetime_utc_merge"].iloc[0] == pd.Timestamp("2023-01-01 08:00:00")

    def test_returns_copy(self, collision_df):
        result = merging._prepare_collision_datetime(collision_df)
        assert result is not collision_df

    @pytest.mark.filterwarnings("ignore:Could not infer format")
    def test_handles_bad_dates_gracefully(self):
        df = pd.DataFrame(
            {
                "date": ["not_a_date", "02/01/2023"],
                "time": [time(8, 30), time(14, 0)],
                "latitude": [51.0, 52.0],
                "longitude": [-1.0, 0.0],
            }
        )
        result = merging._prepare_collision_datetime(df)
        assert pd.isna(result["datetime_local"].iloc[0])
        assert pd.isna(result["datetime_utc"].iloc[0])





class TestPrepareWeatherDatetime:
    def test_converts_to_naive_datetime(self, weather_df):
        result = merging._prepare_weather_datetime(weather_df)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime_utc"])
        assert result["datetime_utc"].dt.tz is None
    def test_returns_copy(self, weather_df):
        result = merging._prepare_weather_datetime(weather_df)
        assert result is not weather_df





class TestMergeDatasets:
    def test_left_merge_shape(self, prepared_collision, prepared_weather):
        result = merging.merge_datasets(prepared_collision, prepared_weather)
        
        assert len(result) == len(prepared_collision)

    def test_matched_rows_have_weather_data(self, prepared_collision, prepared_weather):
        result = merging.merge_datasets(prepared_collision, prepared_weather)
        
        assert result["temperature"].iloc[0] == 5.0
        assert result["humidity"].iloc[0] == 80

    def test_unmatched_rows_are_nan(self, prepared_collision, prepared_weather):
        result = merging.merge_datasets(prepared_collision, prepared_weather)
        
        
        pass

    def test_unmatched_rows_are_nan_custom(self):
        collision = pd.DataFrame(
            {
                "datetime_utc_merge": [pd.Timestamp("2023-01-01 00:00:00")],
                "_lat_r": [99.9],
                "_lon_r": [99.9],
                "collision_id": [1],
            }
        )
        weather = pd.DataFrame(
            {
                "datetime_utc": [pd.Timestamp("2023-01-01 12:00:00")],
                "_lat_r": [51.5],
                "_lon_r": [-0.1],
                "temperature": [10.0],
            }
        )
        result = merging.merge_datasets(collision, weather)
        assert pd.isna(result["temperature"].iloc[0])

    def test_preserves_collision_columns(self, prepared_collision, prepared_weather):
        result = merging.merge_datasets(prepared_collision, prepared_weather)
        assert "collision_id" in result.columns





class TestValidateMerge:
    def test_returns_dict_with_expected_keys(self, prepared_collision, prepared_weather, tmp_path, monkeypatch):
        monkeypatch.setattr(merging.cfg, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(merging.cfg, "MERGING_REPORT_FILE", "merge_report.json")

        merged = merging.merge_datasets(prepared_collision, prepared_weather)
        report = merging.validate_merge(merged, prepared_weather)

        assert isinstance(report, dict)
        assert "merged_rows" in report
        assert "merged_columns" in report
        assert "missing_weather_rows" in report
        assert "weather_merge_rate_pct" in report
        assert "sample_weather_column_checked" in report

    def test_writes_json_file(self, prepared_collision, prepared_weather, tmp_path, monkeypatch):
        monkeypatch.setattr(merging.cfg, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(merging.cfg, "MERGING_REPORT_FILE", "merge_report.json")

        merged = merging.merge_datasets(prepared_collision, prepared_weather)
        merging.validate_merge(merged, prepared_weather)

        report_path = tmp_path / "merge_report.json"
        assert report_path.exists()

        with open(report_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["merged_rows"] == len(merged)

    def test_merge_rate_calculation(self):
        merged = pd.DataFrame(
            {
                "collision_id": [1, 2, 3, 4],
                "temperature": [5.0, np.nan, 7.0, np.nan],
            }
        )
        weather = pd.DataFrame(
            {
                "datetime_utc": [pd.Timestamp("2023-01-01")],
                "_lat_r": [51.5],
                "_lon_r": [-0.1],
                "temperature": [5.0],
            }
        )
        report = merging.validate_merge(merged, weather)
        assert report["missing_weather_rows"] == 2
        assert report["weather_merge_rate_pct"] == 50.0

    def test_no_weather_cols_handled(self):
        merged = pd.DataFrame({"collision_id": [1, 2]})
        weather = pd.DataFrame({"datetime_utc": [1], "_lat_r": [1], "_lon_r": [1]})
        report = merging.validate_merge(merged, weather)
        assert report["weather_merge_rate_pct"] == 0.0
        assert report["sample_weather_column_checked"] is None





class TestSaveMergedData:
    def test_writes_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(merging.cfg, "INTERIM_DATA_DIR", tmp_path)
        monkeypatch.setattr(merging.cfg, "INTERIM_OUTPUT_FILE", "merged.csv")

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        merging.save_merged_data(df)

        out_path = tmp_path / "merged.csv"
        assert out_path.exists()

        loaded = pd.read_csv(out_path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_empty_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(merging.cfg, "INTERIM_DATA_DIR", tmp_path)
        monkeypatch.setattr(merging.cfg, "INTERIM_OUTPUT_FILE", "empty.csv")

        df = pd.DataFrame()
        merging.save_merged_data(df)

        out_path = tmp_path / "empty.csv"
        assert out_path.exists()
        assert out_path.read_text() == "\n"





class TestMergeCollisionWeather:
    @patch.object(merging, "load_raw_data")
    @patch.object(merging, "load_external_data")
    @patch.object(merging, "save_stage_report")
    def test_end_to_end(
        self, mock_save_stage, mock_load_external, mock_load_raw,
        collision_df, weather_df, tmp_path, monkeypatch
    ):
        
        collision_df["date"] = ["01/01/2023", "01/01/2023", "01/01/2023"]
        collision_df["time"] = [time(8, 30), time(8, 45), time(9, 15)]
        collision_df["latitude"] = [51.5074, 51.5074, 51.5074]
        collision_df["longitude"] = [-0.1278, -0.1278, -0.1278]

        mock_load_raw.return_value = collision_df
        mock_load_external.return_value = weather_df

        monkeypatch.setattr(merging.cfg, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(merging.cfg, "MERGING_REPORT_FILE", "report.json")
        monkeypatch.setattr(merging.cfg, "INTERIM_DATA_DIR", tmp_path)
        monkeypatch.setattr(merging.cfg, "INTERIM_OUTPUT_FILE", "merged.csv")

        merging.merge_collision_weather()

        mock_load_raw.assert_called_once()
        mock_load_external.assert_called_once()
        mock_save_stage.assert_called_once_with("merging")

        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "merged.csv").exists()