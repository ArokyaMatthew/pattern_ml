from pathlib import Path

import pandas as pd
import pytest

from arbor.preprocessing import read_csv

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def play_tennis() -> pd.DataFrame:
    return read_csv(DATA / "play_tennis.csv")


@pytest.fixture
def weather() -> pd.DataFrame:
    """Quinlan's golf data: outlook and windy categorical, temperature and humidity numeric."""
    frame = read_csv(DATA / "weather_numeric.csv")
    frame["windy"] = frame["windy"].astype(str)
    return frame


@pytest.fixture
def iris() -> pd.DataFrame:
    return read_csv(DATA / "iris.csv")


@pytest.fixture
def credit() -> pd.DataFrame:
    return read_csv(DATA / "credit_risk.csv")
