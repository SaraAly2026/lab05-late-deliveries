import pandas as pd
import pytest

from latedeliveries.data import make_orders
from latedeliveries.pipeline import split_orders


def test_generated_data_has_3000_rows():
    df = make_orders(n=3000, seed=7)
    assert len(df) == 3000


def test_target_is_binary():
    df = make_orders(n=3000, seed=7)
    assert set(df["late"].unique()).issubset({0, 1})


def test_split_sizes():
    df = make_orders(n=3000, seed=7)
    train, test = split_orders(df)

    assert len(train) == 2250
    assert len(test) == 750


def test_split_is_reproducible():
    df = make_orders(n=3000, seed=7)

    train1, test1 = split_orders(df)
    train2, test2 = split_orders(df)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


@pytest.mark.parametrize("seed", [7, 21, 42])
def test_generated_data_has_expected_columns(seed):
    df = make_orders(n=100, seed=seed)

    assert {
        "distance_km",
        "items",
        "hour",
        "weather",
        "day_type",
        "late",
    }.issubset(df.columns)
