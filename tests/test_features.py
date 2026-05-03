import pytest
import pandas as pd

from src.features.build_features import (
    correlation_based_selection,
    feature_interactions
)


def test_correlation_selection(sample_dataframe):
    X = sample_dataframe.drop(columns=["collision_severity"])
    y = sample_dataframe["collision_severity"]

    X_new, kept, dropped = correlation_based_selection(
        X=X,
        y=y,
        threshold=0.9,
        numerical_cols=list(X.columns)
    )

    assert isinstance(X_new, pd.DataFrame)
    assert len(kept) + len(dropped) >= len(X.columns) - len(dropped)
    assert set(kept).isdisjoint(set(dropped))


def test_feature_interactions_shape_increase(sample_dataframe):
    X = sample_dataframe.drop(columns=["collision_severity"])
    num_cols = ["a"]  # replace with sample fixture or real list
    cat_cols = ["b"]

    X_out, new_num, new_cat = feature_interactions(X, num_cols, cat_cols)

    assert isinstance(X_out, pd.DataFrame)
    assert X_out.shape[1] >= X.shape[1]
    assert isinstance(new_num, list)
    assert isinstance(new_cat, list)


def test_feature_interactions_empty_dataframe(empty_dataframe):
    num_cols = []
    cat_cols = []

    result = feature_interactions(empty_dataframe, num_cols, cat_cols)

    assert result is None


@pytest.mark.parametrize(
    "expected_col",
    [
        "casualties_per_vehicle",
        "rain_intensity",
        "wind_force",
        "visibility_risk",
        "is_bad_weather",
    ]
)
def test_feature_interactions_creates_expected_columns(sample_dataframe, expected_col):
    X = sample_dataframe.drop(columns=["collision_severity"])
    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = []

    X_out, _, _ = feature_interactions(X, num_cols, cat_cols)

    assert expected_col in X_out.columns


def test_correlation_threshold_effect(sample_dataframe):
    X = sample_dataframe.drop(columns=["collision_severity"])
    y = sample_dataframe["collision_severity"]

    _, kept_low, _ = correlation_based_selection(X, y, 0.5, list(X.columns))
    _, kept_high, _ = correlation_based_selection(X, y, 0.9, list(X.columns))

    assert len(kept_low) <= len(kept_high)