from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import src.data.cleaning as cleaning


@pytest.fixture(autouse=True)
def suppress_log_action(monkeypatch):
    """Prevent side-effects from ``log_action`` during unit tests."""
    monkeypatch.setattr(cleaning, "log_action", lambda **kwargs: None)


@pytest.fixture
def basic_cleaning_df() -> pd.DataFrame:
    """Minimal valid DataFrame that survives all cleaning steps."""
    return pd.DataFrame(
        {
            "longitude": [-2.0, -1.0, 0.0],
            "latitude": [51.0, 52.0, 53.0],
            "number_of_vehicles": [2, 3, 1],
            "number_of_casualties": [1, 2, 1],
            "weather_conditions": [1, 2, 3],
            "wx_cloud_cover": [1, 2, 1],
            "collision_severity": [1, 2, 3],
        }
    )


@pytest.fixture
def dirty_geo_df() -> pd.DataFrame:
    """Rows with invalid geography or non-positive counts."""
    return pd.DataFrame(
        {
            "latitude": [51.5, 70.0, 49.0, 55.0, 60.0],
            "longitude": [-2.0, -1.0, -10.0, 2.0, 0.0],
            "number_of_vehicles": [2, 3, 1, 0, 2],
            "number_of_casualties": [1, 0, 2, 1, 2],
        }
    )


@pytest.fixture
def coded_missing_df() -> pd.DataFrame:
    """Sentinel codes that should become NaN."""
    return pd.DataFrame(
        {
            "weather_conditions": [1, 2, -1, 4, -1],
            "wx_cloud_cover": [1, 99, 2, 99, 1],
            "longitude": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def completeness_df() -> pd.DataFrame:
    """Assorted missing-value patterns for completeness testing."""
    return pd.DataFrame(
        {
            "num_a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "num_b": [np.nan, np.nan, np.nan, 4.0, 5.0],
            "cat_a": pd.Categorical(["x", "y", np.nan, "x", "y"]),
            "cat_b": pd.Categorical([np.nan, np.nan, np.nan, "a", "b"]),
            "mcar_col": [1.0, np.nan, 3.0, 4.0, 5.0],
            "target": [0, 1, 2, 0, 1],
        }
    )


class TestDropUnwantedColumns:
    def test_drops_present_columns(self, monkeypatch, basic_cleaning_df):
        monkeypatch.setattr(cleaning.cfg, "COLUMNS_TO_DROP", ["weather_conditions"])
        result = cleaning.drop_unwanted_columns(basic_cleaning_df)
        assert "weather_conditions" not in result.columns
        assert "longitude" in result.columns

    def test_ignores_missing_columns(self, monkeypatch, basic_cleaning_df):
        monkeypatch.setattr(cleaning.cfg, "COLUMNS_TO_DROP", ["not_a_column"])
        result = cleaning.drop_unwanted_columns(basic_cleaning_df)
        assert list(result.columns) == list(basic_cleaning_df.columns)

    def test_returns_copy(self, monkeypatch, basic_cleaning_df):
        monkeypatch.setattr(cleaning.cfg, "COLUMNS_TO_DROP", [])
        result = cleaning.drop_unwanted_columns(basic_cleaning_df)
        assert result is not basic_cleaning_df

    def test_empty_dataframe(self, monkeypatch, empty_dataframe):
        monkeypatch.setattr(cleaning.cfg, "COLUMNS_TO_DROP", ["foo"])
        result = cleaning.drop_unwanted_columns(empty_dataframe)
        assert result.empty


# ----------------------------------------------------------------------
# 2. cast_column_types
# ----------------------------------------------------------------------
class TestCastColumnTypes:
    def test_casts_int_columns_to_category(self, monkeypatch, basic_cleaning_df):
        monkeypatch.setattr(
            cleaning.cfg, "CATEGORICAL_COLS", ["weather_conditions", "wx_cloud_cover"]
        )
        result = cleaning.cast_column_types(basic_cleaning_df)
        assert result["weather_conditions"].dtype == "category"
        assert result["wx_cloud_cover"].dtype == "category"

    def test_parses_time_column(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "CATEGORICAL_COLS", [])
        df = pd.DataFrame({"time": ["08:30", "12:45", "bad_time", "23:59"]})
        result = cleaning.cast_column_types(df)
        assert result["time"].iloc[0] == pd.to_datetime("08:30", format="%H:%M").time()
        assert pd.isna(result["time"].iloc[2])

    def test_no_error_when_time_missing(self, monkeypatch, basic_cleaning_df):
        monkeypatch.setattr(cleaning.cfg, "CATEGORICAL_COLS", [])
        result = cleaning.cast_column_types(basic_cleaning_df)
        assert "time" not in result.columns


# ----------------------------------------------------------------------
# 3. validate_accuracy
# ----------------------------------------------------------------------
class TestValidateAccuracy:
    def test_quarantines_invalid_geo(self, monkeypatch, dirty_geo_df):
        monkeypatch.setattr(cleaning.cfg, "UK_LAT_MIN", 49.9)
        monkeypatch.setattr(cleaning.cfg, "UK_LAT_MAX", 60.9)
        monkeypatch.setattr(cleaning.cfg, "UK_LON_MIN", -8.6)
        monkeypatch.setattr(cleaning.cfg, "UK_LON_MAX", 1.8)

        def _mock_quarantine(df, mask, reason, stage):
            return df[~mask].copy()

        monkeypatch.setattr(cleaning, "quarantine", _mock_quarantine)

        result = cleaning.validate_accuracy(dirty_geo_df)
        # Rows 0 & 4 are the only ones fully valid after both geo + count checks
        assert result.shape[0] == 2

    def test_quarantines_non_positive_counts(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "UK_LAT_MIN", 40.0)
        monkeypatch.setattr(cleaning.cfg, "UK_LAT_MAX", 70.0)
        monkeypatch.setattr(cleaning.cfg, "UK_LON_MIN", -20.0)
        monkeypatch.setattr(cleaning.cfg, "UK_LON_MAX", 20.0)

        def _mock_quarantine(df, mask, reason, stage):
            return df[~mask].copy()

        monkeypatch.setattr(cleaning, "quarantine", _mock_quarantine)

        df = pd.DataFrame(
            {
                "latitude": [51.0, 52.0, 53.0],
                "longitude": [0.0, 1.0, 2.0],
                "number_of_vehicles": [2, 0, 1],
                "number_of_casualties": [1, 2, 0],
            }
        )
        result = cleaning.validate_accuracy(df)
        assert False


# ----------------------------------------------------------------------
# 4. fix_consistency
# ----------------------------------------------------------------------
class TestFixConsistency:
    def test_replaces_coded_missing(self, monkeypatch, coded_missing_df):
        monkeypatch.setattr(
            cleaning.cfg,
            "CODED_MISSING",
            {"weather_conditions": [-1], "wx_cloud_cover": [-1, 99]},
        )
        result = cleaning.fix_consistency(coded_missing_df)
        assert result["weather_conditions"].isna().sum() == 2
        assert result["wx_cloud_cover"].isna().sum() == 2
        assert result["longitude"].notna().all()

    def test_skips_missing_columns(self, monkeypatch, coded_missing_df):
        monkeypatch.setattr(cleaning.cfg, "CODED_MISSING", {"nonexistent": [-1]})
        result = cleaning.fix_consistency(coded_missing_df)
        # Should not raise and should leave everything intact
        assert result["weather_conditions"].isna().sum() == 0


# ----------------------------------------------------------------------
# 5. handle_completeness
# ----------------------------------------------------------------------
class TestHandleCompleteness:
    def test_fit_drops_high_null_columns(self, monkeypatch, completeness_df):
        monkeypatch.setattr(cleaning.cfg, "COLS_NULL_PREC", 0.5)
        monkeypatch.setattr(cleaning.cfg, "MCAR_COLS", [])
        monkeypatch.setattr(cleaning.cfg, "ROW_NULL_PREC", 0.1)

        result_df, params = cleaning.handle_completeness(completeness_df, params=None)

        assert "num_b" not in result_df.columns
        assert "cat_b" not in result_df.columns
        assert set(params["cols_to_drop"]) == {"num_b", "cat_b"}

    def test_fit_drops_mcar_rows(self, monkeypatch, completeness_df):
        monkeypatch.setattr(cleaning.cfg, "COLS_NULL_PREC", 0.5)
        monkeypatch.setattr(cleaning.cfg, "MCAR_COLS", ["mcar_col"])
        monkeypatch.setattr(cleaning.cfg, "ROW_NULL_PREC", 0.3)

        result_df, params = cleaning.handle_completeness(completeness_df, params=None)

        assert params["mcar_cols"] == ["mcar_col"]
        # Row 1 has mcar_col missing -> dropped
        assert result_df.shape[0] == 4
        assert result_df["mcar_col"].notna().all()

    def test_fit_imputes_numerical_median(self, monkeypatch, completeness_df):
        monkeypatch.setattr(cleaning.cfg, "COLS_NULL_PREC", 0.5)
        monkeypatch.setattr(cleaning.cfg, "MCAR_COLS", [])
        monkeypatch.setattr(cleaning.cfg, "ROW_NULL_PREC", 0.1)

        result_df, params = cleaning.handle_completeness(completeness_df, params=None)

        assert result_df["num_a"].isna().sum() == 0
        # median of [1, 2, 4, 5] (ignoring NaN) == 3.0
        assert params["numerical_impute"]["num_a"] == 3.0

    def test_fit_imputes_categorical_mode(self, monkeypatch, completeness_df):
        monkeypatch.setattr(cleaning.cfg, "COLS_NULL_PREC", 0.5)
        monkeypatch.setattr(cleaning.cfg, "MCAR_COLS", [])
        monkeypatch.setattr(cleaning.cfg, "ROW_NULL_PREC", 0.1)

        result_df, params = cleaning.handle_completeness(completeness_df, params=None)

        assert result_df["cat_a"].isna().sum() == 0
        # mode of ["x", "y", "x", "y"] resolves to "x" (sorted tie-break)
        assert params["categorical_impute"]["cat_a"] == "x"

    def test_transform_uses_provided_params(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "COLS_NULL_PREC", 0.5)
        monkeypatch.setattr(cleaning.cfg, "MCAR_COLS", ["mcar_col"])
        monkeypatch.setattr(cleaning.cfg, "ROW_NULL_PREC", 0.3)

        params = {
            "cols_to_drop": ["num_b"],
            "mcar_cols": ["mcar_col"],
            "numerical_impute": {"num_a": 99.0},
            "categorical_impute": {"cat_a": "Z"},
        }

        df = pd.DataFrame(
            {
                "num_a": [np.nan, 20.0, 30.0],
                "num_b": [1.0, 2.0, 3.0],
                "cat_a": pd.Categorical([np.nan, "y", "y"], categories=["y", "Z"]),
                "mcar_col": [1.0, np.nan, 2.0],
                "target": [0, 1, 0],
            }
        )

        result_df, _ = cleaning.handle_completeness(df, params=params)

        assert "num_b" not in result_df.columns
        assert result_df.loc[0, "num_a"] == 99.0
        assert result_df.loc[0, "cat_a"] == "Z"
        assert result_df.shape[0] == 2
        assert result_df.loc[2, "num_a"] == 30.0


# ----------------------------------------------------------------------
# 6. handle_outliers
# ----------------------------------------------------------------------
class TestHandleOutliers:
    def test_skips_lat_lon(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "DOMINANT_THRESHOLD", 0.7)
        monkeypatch.setattr(cleaning.cfg, "OUTLIER_RATIO_THRESHOLD", 0.05)

        df = pd.DataFrame(
            {
                "longitude": [0.0, 999.0, -999.0],
                "latitude": [51.0, 500.0, 600.0],
                "other": [1, 2, 3],
            }
        )

        _, params = cleaning.handle_outliers(df, params=None)
        assert "longitude" not in params
        assert "latitude" not in params
        assert "other" in params

    def test_fit_iqr_capping(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "DOMINANT_THRESHOLD", 0.7)
        monkeypatch.setattr(cleaning.cfg, "OUTLIER_RATIO_THRESHOLD", 0.05)

        df = pd.DataFrame(
            {
                "value": list(range(1, 11)) + [100],  # 100 is extreme
                "longitude": [0.0] * 11,
                "latitude": [51.0] * 11,
            }
        )

        result_df, params = cleaning.handle_outliers(df, params=None)
        col_params = params["value"]
        assert col_params["strategy"] == "iqr"
        # 1 outlier / 11 rows ≈ 9 % > 5 %  → cap rather than remove
        assert col_params["action"] == "cap"
        assert result_df["value"].max() <= col_params["upper"]

    def test_fit_iqr_removal(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "DOMINANT_THRESHOLD", 0.7)
        monkeypatch.setattr(cleaning.cfg, "OUTLIER_RATIO_THRESHOLD", 0.5)

        df = pd.DataFrame(
            {
                "value": list(range(1, 11)) + [100],
                "longitude": [0.0] * 11,
                "latitude": [51.0] * 11,
            }
        )

        result_df, params = cleaning.handle_outliers(df, params=None)
        assert params["value"]["action"] == "remove"
        assert 100 not in result_df["value"].values

    def test_fit_mode_heavy(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "DOMINANT_THRESHOLD", 0.7)
        monkeypatch.setattr(cleaning.cfg, "OUTLIER_RATIO_THRESHOLD", 0.05)

        df = pd.DataFrame(
            {
                "mode_col": [1] * 90 + list(range(2, 12)),
                "longitude": [0.0] * 100,
                "latitude": [51.0] * 100,
            }
        )

        _, params = cleaning.handle_outliers(df, params=None)
        col_params = params["mode_col"]
        assert col_params["strategy"] == "mode_heavy"
        assert col_params["dominant_val"] == 1
        assert "lower" in col_params
        assert "upper" in col_params

    def test_transform_applies_stored_params(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "DOMINANT_THRESHOLD", 0.7)
        monkeypatch.setattr(cleaning.cfg, "OUTLIER_RATIO_THRESHOLD", 0.05)

        params = {
            "value": {
                "strategy": "iqr",
                "lower": 0.0,
                "upper": 10.0,
                "action": "cap",
            }
        }

        df = pd.DataFrame(
            {
                "value": [5.0, 15.0, -5.0],
                "longitude": [0.0] * 3,
                "latitude": [51.0] * 3,
            }
        )

        result_df, _ = cleaning.handle_outliers(df, params=params)
        assert result_df["value"].iloc[1] == 10.0
        assert result_df["value"].iloc[2] == 0.0

    def test_transform_skips_none_action(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "DOMINANT_THRESHOLD", 0.7)
        monkeypatch.setattr(cleaning.cfg, "OUTLIER_RATIO_THRESHOLD", 0.05)

        params = {
            "value": {
                "strategy": "iqr",
                "lower": 0.0,
                "upper": 100.0,
                "action": "none",
            }
        }

        df = pd.DataFrame(
            {
                "value": [5.0, 200.0],
                "longitude": [0.0] * 2,
                "latitude": [51.0] * 2,
            }
        )

        result_df, _ = cleaning.handle_outliers(df, params=params)
        assert result_df["value"].iloc[1] == 200.0


# ----------------------------------------------------------------------
# 7. invert_severity_labels
# ----------------------------------------------------------------------
class TestInvertSeverityLabels:
    def test_inverts_labels(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "TARGET_COL", "collision_severity")
        df = pd.DataFrame({"collision_severity": [1, 2, 3, 1, 3]})
        result = cleaning.invert_severity_labels(df)
        expected = pd.Series([2, 1, 0, 2, 0], name="collision_severity")
        pd.testing.assert_series_equal(result["collision_severity"], expected)

    def test_returns_copy(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "TARGET_COL", "collision_severity")
        df = pd.DataFrame({"collision_severity": [1, 2, 3]})
        result = cleaning.invert_severity_labels(df)
        assert result is not df


# ----------------------------------------------------------------------
# 8. split_dataset
# ----------------------------------------------------------------------
class TestSplitDataset:
    def test_stratified_split(self, monkeypatch):
        monkeypatch.setattr(cleaning.cfg, "TEST_SIZE", 0.2)
        monkeypatch.setattr(cleaning.cfg, "RANDOM_STATE", 42)
        monkeypatch.setattr(cleaning.cfg, "TARGET_COL", "collision_severity")

        df = pd.DataFrame(
            {
                "feat": range(100),
                "collision_severity": [0] * 50 + [1] * 30 + [2] * 20,
            }
        )

        train, test = cleaning.split_dataset(df)
        assert len(train) == 80
        assert len(test) == 20
        assert set(train["collision_severity"].unique()) == {0, 1, 2}
        assert set(test["collision_severity"].unique()) == {0, 1, 2}


# ----------------------------------------------------------------------
# 9. build_clean_dataset  (integration)
# ----------------------------------------------------------------------
class TestBuildCleanDataset:
    @patch.object(cleaning, "load_csv")
    @patch.object(cleaning, "quarantine")
    @patch.object(cleaning, "save_stage_report")
    def test_end_to_end_pipeline(
        self, mock_save, mock_quarantine, mock_load, tmp_path, monkeypatch
    ):
        mock_df = pd.DataFrame(
            {
                "admin_col": range(15),
                "longitude": [-2.0, -1.0, 0.0, 0.5, -3.0] * 3,
                "latitude": [51.0, 52.0, 53.0, 54.0, 55.0] * 3,
                "number_of_vehicles": [2, 3, 1, 2, 1] * 3,
                "number_of_casualties": [1, 2, 1, 1, 2] * 3,
                "weather_conditions": [1, 2, 3, 1, 2] * 3,
                "collision_severity": [1, 2, 3, 1, 2] * 3,
            }
        )
        mock_load.return_value = mock_df

        def _mock_q(df, mask, reason, stage):
            return df[~mask].copy()

        mock_quarantine.side_effect = _mock_q

        monkeypatch.setattr(cleaning.cfg, "COLUMNS_TO_DROP", ["admin_col"])
        monkeypatch.setattr(cleaning.cfg, "CATEGORICAL_COLS", ["weather_conditions"])
        monkeypatch.setattr(cleaning.cfg, "UK_LAT_MIN", 49.9)
        monkeypatch.setattr(cleaning.cfg, "UK_LAT_MAX", 60.9)
        monkeypatch.setattr(cleaning.cfg, "UK_LON_MIN", -8.6)
        monkeypatch.setattr(cleaning.cfg, "UK_LON_MAX", 1.8)
        monkeypatch.setattr(cleaning.cfg, "CODED_MISSING", {})
        monkeypatch.setattr(cleaning.cfg, "COLS_NULL_PREC", 0.5)
        monkeypatch.setattr(cleaning.cfg, "MCAR_COLS", [])
        monkeypatch.setattr(cleaning.cfg, "ROW_NULL_PREC", 0.1)
        monkeypatch.setattr(cleaning.cfg, "DOMINANT_THRESHOLD", 0.7)
        monkeypatch.setattr(cleaning.cfg, "OUTLIER_RATIO_THRESHOLD", 0.05)
        monkeypatch.setattr(cleaning.cfg, "TEST_SIZE", 0.2)
        monkeypatch.setattr(cleaning.cfg, "RANDOM_STATE", 42)
        monkeypatch.setattr(cleaning.cfg, "TARGET_COL", "collision_severity")
        monkeypatch.setattr(cleaning.cfg, "INTERIM_DATA_DIR", tmp_path)
        monkeypatch.setattr(cleaning.cfg, "CLEANED_TRAIN_OUTPUT_FILE", "train.csv")
        monkeypatch.setattr(cleaning.cfg, "CLEANED_TEST_OUTPUT_FILE", "test.csv")

        train, test = cleaning.build_clean_dataset("dummy_path.csv")

        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert "admin_col" not in train.columns
        assert "admin_col" not in test.columns
        mock_load.assert_called_once_with("dummy_path.csv")
        mock_save.assert_called_once_with("cleaning")
        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "test.csv").exists()