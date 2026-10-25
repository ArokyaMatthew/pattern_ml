"""Regenerate the sample datasets in ``data/``.

Run from the repository root:

    python scripts/make_datasets.py

The UCI datasets come from scikit-learn's bundled copies, so no download
is needed. ``credit_risk.csv`` is synthetic and fully reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

DATA = Path(__file__).resolve().parent.parent / "data"

PLAY_TENNIS = """\
sunny,hot,high,weak,no
sunny,hot,high,strong,no
overcast,hot,high,weak,yes
rain,mild,high,weak,yes
rain,cool,normal,weak,yes
rain,cool,normal,strong,no
overcast,cool,normal,strong,yes
sunny,mild,high,weak,no
sunny,cool,normal,weak,yes
rain,mild,normal,weak,yes
sunny,mild,normal,strong,yes
overcast,mild,high,strong,yes
overcast,hot,normal,weak,yes
rain,mild,high,strong,no"""

# Quinlan (1993), table 1-1; also Weka's weather.numeric.arff.
WEATHER_NUMERIC = """\
sunny,85,85,false,no
sunny,80,90,true,no
overcast,83,86,false,yes
rainy,70,96,false,yes
rainy,68,80,false,yes
rainy,65,70,true,no
overcast,64,65,true,yes
sunny,72,95,false,no
sunny,69,70,false,yes
rainy,75,80,false,yes
sunny,75,70,true,yes
overcast,72,90,true,yes
overcast,81,75,false,yes
rainy,71,91,true,no"""


def from_rows(text: str, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([line.split(",") for line in text.splitlines()], columns=columns)


def from_sklearn(bunch, target: str, labels: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(bunch.data, columns=[c.replace(" ", "_").replace("/", "_").replace("(cm)", "").strip("_") for c in bunch.feature_names])
    frame.columns = [c.rstrip("_") for c in frame.columns]
    frame[target] = [labels[i] for i in bunch.target]
    return frame


def credit_risk(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Synthetic loan applications with a noisy, partly non-linear risk rule."""
    rng = np.random.default_rng(seed)
    age = rng.integers(19, 70, n)
    income = (rng.lognormal(mean=10.55, sigma=0.45, size=n)).round(-2)
    employment = rng.choice(["salaried", "self-employed", "contract", "unemployed"], n, p=[0.55, 0.2, 0.17, 0.08])
    history = rng.choice(["good", "fair", "poor"], n, p=[0.5, 0.32, 0.18])
    housing = rng.choice(["own", "mortgage", "rent"], n, p=[0.25, 0.35, 0.4])
    purpose = rng.choice(["car", "education", "home improvement", "business", "personal"], n)
    loan_amount = (rng.gamma(2.2, 4500, n).round(-2) + 500).astype(int)
    term = rng.choice([12, 24, 36, 48, 60], n)
    existing_loans = rng.poisson(0.8, n)

    burden = loan_amount / income
    score = (
        2.0 * (history == "poor")
        + 0.9 * (history == "fair")
        + 1.7 * (employment == "unemployed")
        + 0.6 * (employment == "contract")
        + 0.3 * (employment == "self-employed")
        + 3.2 * np.clip(burden - 0.15, 0, None)
        + 0.35 * existing_loans
        + 0.012 * (term - 12)
        - 0.02 * (age - 19)
        + 0.35 * (housing == "rent")
        + 0.4 * (purpose == "business")
        + rng.normal(0, 0.55, n)
    )
    risk = np.where(score > np.quantile(score, 0.64), "high", "low")

    frame = pd.DataFrame(
        {
            "age": age,
            "annual_income": income,
            "employment": employment,
            "credit_history": history,
            "housing": housing,
            "purpose": purpose,
            "loan_amount": loan_amount,
            "term_months": term,
            "existing_loans": existing_loans,
            "risk": risk,
        }
    )
    for column, rate in {"annual_income": 0.05, "employment": 0.04, "credit_history": 0.03, "housing": 0.02}.items():
        frame[column] = frame[column].astype(object)
        frame.loc[rng.random(n) < rate, column] = np.nan
    return frame


def main() -> None:
    DATA.mkdir(exist_ok=True)
    datasets = {
        "play_tennis.csv": from_rows(PLAY_TENNIS, ["outlook", "temperature", "humidity", "wind", "play"]),
        "weather_numeric.csv": from_rows(WEATHER_NUMERIC, ["outlook", "temperature", "humidity", "windy", "play"]),
        "iris.csv": from_sklearn(load_iris(), "species", ["setosa", "versicolor", "virginica"]),
        "wine.csv": from_sklearn(load_wine(), "cultivar", ["cultivar_a", "cultivar_b", "cultivar_c"]),
        "breast_cancer.csv": from_sklearn(load_breast_cancer(), "diagnosis", ["malignant", "benign"]),
        "credit_risk.csv": credit_risk(),
    }
    for name, frame in datasets.items():
        frame.to_csv(DATA / name, index=False)
        print(f"{name:<22} {frame.shape[0]:>4} rows  {frame.shape[1] - 1:>2} attributes")


if __name__ == "__main__":
    main()
