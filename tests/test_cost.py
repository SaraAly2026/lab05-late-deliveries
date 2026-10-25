import pytest

from latedeliveries.cost import business_cost


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
