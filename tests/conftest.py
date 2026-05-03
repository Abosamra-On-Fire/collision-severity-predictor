import pytest
import pandas as pd
import numpy as np

from src.utils import log_action 


@pytest.fixture(autouse=True)
def suppress_log_action(monkeypatch):
    """
    Globally suppress ``log_action`` so no test writes to the database,
    filesystem, or external log store.
    """
    monkeypatch.setattr(
        "src.utils.log_action",
        lambda **kwargs: None,
    )

@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    df = pd.DataFrame({
        "longitude": np.random.rand(100),
        "latitude": np.random.rand(100),
        "number_of_vehicles": np.random.randint(1, 5, 100),
        "number_of_casualties": np.random.randint(0, 3, 100),
        "wx_rain": np.random.rand(100),
        "wx_precipitation": np.random.rand(100),
        "wx_wind_speed_10m": np.random.rand(100),
        "wx_wind_gusts_10m": np.random.rand(100),
        "weather_conditions": np.random.randint(1, 10, 100),
        "collision_severity": np.random.randint(1, 4, 100),
        "wx_cloud_cover": np.random.randint(1, 4, 100),
    })
    return df

@pytest.fixture
def empty_dataframe():
    return pd.DataFrame()