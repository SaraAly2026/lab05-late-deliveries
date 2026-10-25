from unittest.mock import patch

import numpy as np
import pytest

from latedeliveries.cost import business_cost, optimize_threshold


@pytest.mark.parametrize(
    ("actual", "predicted", "expected"),
    [
        ([0, 0, 0], [0, 0, 0], 0),
        ([1, 1, 1], [0, 0, 0], 45),
        ([0, 0, 0], [1, 1, 1], 12),
        ([1, 0, 1, 0], [1, 1, 0, 0], 23),
        ([1, 1], [1, 1], 8),
    ],
)
def test_business_cost(actual, predicted, expected):
    assert business_cost(actual, predicted) == expected


def test_threshold_can_choose_flag_nobody():
    y_true = np.array([0, 0, 0, 0, 1])
    probabilities = np.array([0.10, 0.20, 0.30, 0.40, 0.01])

    with patch(
        "latedeliveries.cost.cross_val_predict",
        return_value=np.column_stack((1 - probabilities, probabilities)),
    ):
        result = optimize_threshold(None, None, y_true)

    assert result["best_cost"] == 15
    assert result["best_threshold"] == 0.45


def test_threshold_can_choose_flag_everybody():
    y_true = np.array([1, 1, 1, 1, 1])
    probabilities = np.array([0.80, 0.70, 0.60, 0.50, 0.40])

    with patch(
        "latedeliveries.cost.cross_val_predict",
        return_value=np.column_stack((1 - probabilities, probabilities)),
    ):
        result = optimize_threshold(None, None, y_true)

    assert result["best_cost"] == 20
    assert result["best_threshold"] == 0.05
