import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import build_pipeline


def test_model_reload(tmp_path):
    df = make_orders(n=200, seed=7)
    X = df.drop(columns=["late"])
    y = df["late"]

    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    loaded_model = joblib.load(model_path)

    original = model.predict_proba(X)[:, 1]
    reloaded = loaded_model.predict_proba(X)[:, 1]

    np.testing.assert_allclose(original, reloaded)


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.20, 0.25, 0),
        (0.25, 0.25, 1),
        (0.80, 0.25, 1),
        (0.10, 0.50, 0),
    ],
)
def test_threshold_decision(probability, threshold, expected):
    prediction = int(probability >= threshold)
    assert prediction == expected
