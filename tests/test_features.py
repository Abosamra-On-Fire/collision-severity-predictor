import pytest
from src.features.build_features import correlation_based_selection, feature_interactions


def test_correlation_selection(sample_dataframe):
    X = sample_dataframe.drop(columns=["collision_severity"])
    y = sample_dataframe["collision_severity"]

    X_new, kept, dropped = correlation_based_selection(
        X=X,
        y=y,
        threshold=0.9,
        numerical_cols=list(X.columns)
    )

    assert isinstance(X_new, type(X))
    assert len(kept) + len(dropped) == len(X.columns)
    # at2aked en pd.DataFrame byrg3 we magmo3 el columns consistent


def test_no_columns_lost(sample_dataframe):
    X = sample_dataframe.drop(columns=["collision_severity"])
    y = sample_dataframe["collision_severity"]
    X_out = feature_interactions(X)
    assert X_out.shape[1] >= X.shape[1]  
    # at2aked en el feature interactions btzawed columns



def test_empty_dataframe_behavior(empty_dataframe):
    try:
        feature_interactions(empty_dataframe)
    except Exception as e:
        assert False, f"Should not crash: {e}"
    # Lw empty dataframe passed el mfrod my7salsh crash




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

    X_out = feature_interactions(X)

    assert expected_col in X_out.columns



def test_correlation_threshold_effect(sample_dataframe):
    X = sample_dataframe.drop(columns=["collision_severity"])
    y = sample_dataframe["collision_severity"]

    _, kept_low, _ = correlation_based_selection(X, y, 0.5, list(X.columns))
    _, kept_high, _ = correlation_based_selection(X, y, 0.9, list(X.columns))

    assert len(kept_low) <= len(kept_high)
    # lma el threshold y2el el mfrod columns aktr ttshal, fa kept_high el mfrod tb2a akbar aw ad kept_low