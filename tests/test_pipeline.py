import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
)


def test_features_exclude_leakage():
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert "prep_minutes" not in features
    assert "customer_rating" not in features
    assert "order_id" not in features
    assert len(features) == 5


def test_probabilities_are_valid():
    df = make_orders(n=300, seed=7)

    X = df.drop(columns=["late"])
    y = df["late"]

    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]

    assert len(probabilities) == len(X)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_missing_values_and_unseen_categories():
    df = make_orders(n=300, seed=7)

    X = df.drop(columns=["late"])
    y = df["late"]

    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X, y)

    new_order = pd.DataFrame(
        [
            {
                "distance_km": np.nan,
                "items": 3,
                "hour": 18,
                "weather": "unknown_weather",
                "day_type": "unknown_day",
            }
        ]
    )

    probabilities = model.predict_proba(new_order)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
