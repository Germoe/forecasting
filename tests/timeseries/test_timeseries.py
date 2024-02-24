import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecasting.timeseries import TimeSeries

from functools import reduce

ts_multi_idx_dummy_path = "./tests/timeseries/ts_multi_idx_dummy.json"
ts_multi_col_multi_idx_dummy_path = (
    "./tests/timeseries/ts_multi_col_multi_idx_dummy.json"
)


@pytest.fixture
def ts_data():
    df = pd.read_json(ts_multi_idx_dummy_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date"], inplace=True)
    df.index = df.index.to_period("D")
    df.set_index(["family", "store_nbr"], append=True, inplace=True)

    return df


@pytest.fixture
def ts_multi_col_data():
    df = pd.read_json(ts_multi_col_multi_idx_dummy_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date"], inplace=True)
    df.index = df.index.to_period("D")
    df.set_index(["family", "store_nbr"], append=True, inplace=True)

    return df


def test_ts_init(ts_data):
    ts = TimeSeries(ts_data)
    assert ts.ts.equals(ts_data.sort_index())


def test_ts_init_no_data():
    """
    Test that an empty DataFrame is not allowed.
    """
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D").to_period("D")
    families = ["family_1"] * 10
    idx = pd.MultiIndex.from_product([dates, families], names=["date", "family"])
    df = pd.DataFrame([], index=idx)

    with pytest.raises(ValueError) as exp:
        TimeSeries(df)

    assert str(exp.value) == "Empty DataFrame is not allowed."


def test_ts_repr(ts_data):
    ts = TimeSeries(ts_data)
    assert repr(ts) == ts_data.sort_index().head().__repr__()


def test_periodindex_exists():
    """
    Test that the DataFrame has a PeriodIndex at level 0.
    """
    # Load data without a PeriodIndex
    df = pd.read_json(ts_multi_idx_dummy_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date", "family", "store_nbr"], inplace=True)

    with pytest.raises(IndexError) as exp:
        TimeSeries(df)

    assert str(exp.value) == "The DataFrame must have a PeriodIndex at level 0."


def test_ts_get_timestamp_index(ts_data):
    """
    Test the conversion of a PeriodIndex to a TimestampIndex.
    """
    ts = TimeSeries(ts_data)
    ts_dt = ts._get_timestamp_index()
    assert isinstance(ts_dt, pd.DataFrame)
    assert isinstance(ts_dt.index.levels[0], pd.DatetimeIndex)


def test_ts_plot_default(ts_data):
    """
    Tests a plotting method but avoids visual comparisons and uses
    data comparison instead.
    """
    ts = TimeSeries(ts_data)

    # Select the number of unique index combinations (excl. date index)
    idx_len = len(ts_data.droplevel(0).index.unique())

    # Test the default plot
    axes = ts.plot()
    assert isinstance(axes[0], plt.Axes)
    assert len(axes.flatten()) == idx_len

    fig = plt.gcf()
    assert fig.get_figwidth() == 12
    assert fig.get_figheight() == 3 * idx_len

    # Check that x-axis is a DatetimeIndex
    assert isinstance(axes[0].lines[0].get_xdata()[0], np.datetime64)


def test_ts_plot_multi_col(ts_multi_col_data):
    ts = TimeSeries(ts_multi_col_data)

    # Select the number of unique index combinations (excl. date index)
    idx_len = len(ts_multi_col_data.droplevel(0).index.unique())
    col_len = len(ts_multi_col_data.columns)

    # Test the default plot with multiple columns
    axes = ts.plot()
    assert (
        reduce(lambda x, y: x + y, [len(ax.get_lines()) for ax in axes])
        == idx_len * col_len
    )

    # Test the plot with a subset of columns
    axes = ts.plot(columns=["sales"])
    assert (
        reduce(lambda x, y: x + y, [len(ax.get_lines()) for ax in axes]) == idx_len * 1
    )


def test_ts_plot_subset(ts_data):
    ts = TimeSeries(ts_data)

    ax = ts.plot(index=(slice(None), "CLEANING", 1))
    assert isinstance(ax, plt.Axes)
    assert len(ax.get_lines()) == 1
    assert ax.lines[0].get_xdata().shape == (
        len(ts_data.loc[(slice(None), "CLEANING", 1), :].index),
    )

    axes = ts.plot(index=(slice(None), slice(None), 1))
    assert len(axes) == 2
    assert reduce(lambda x, y: x + y, [len(ax.get_lines()) for ax in axes]) == 2


def test_ts_large_nr_plot():
    # Generate a large number of index combinations
    n = 50
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D").to_period("D")
    families = [f"family_{i}" for i in range(n)]
    stores = [f"store_{i}" for i in range(n)]
    idx = pd.MultiIndex.from_product(
        [dates, families, stores], names=["date", "family", "store_nbr"]
    )
    values = {"sales": [i for i in range(n * n * n)]}
    df = pd.DataFrame(values, index=idx)

    ts = TimeSeries(df)

    with pytest.raises(ValueError) as exp:
        ts.plot()

    assert (
        str(exp.value)
        == "The number of index combinations is too large to plot individually."
    )