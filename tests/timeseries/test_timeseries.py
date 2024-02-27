import pytest

import numpy as np
import pandas as pd
import io
from matplotlib.collections import PathCollection
from matplotlib.figure import SubFigure
import matplotlib.pyplot as plt
from forecasting.timeseries import TimeSeries

from functools import reduce

ts_multi_idx_dummy_path = "./tests/timeseries/ts_multi_idx_dummy.json"
ts_multi_col_multi_idx_dummy_path = (
    "./tests/timeseries/ts_multi_col_multi_idx_dummy.json"
)
ts_corr_dummy_path = "./tests/timeseries/ts_corr_dummy.json"


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


@pytest.fixture
def ts_corr_dummy_data():
    df = pd.read_json(ts_corr_dummy_path)
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

    # Check Line color
    assert axes[0].lines[0].get_color() == "black"
    assert axes[0].lines[0].get_linewidth() == 0.5

    # Check that x-axis is a DatetimeIndex
    assert isinstance(axes[0].lines[0].get_xdata()[0], np.datetime64)


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

    axes = ts.plot(index=(slice(None), "CLEANING", 1))
    assert isinstance(axes[0], plt.Axes)
    assert len(axes[0].get_lines()) == 1
    assert axes[0].lines[0].get_xdata().shape == (
        len(ts_data.loc[(slice(None), "CLEANING", 1), :].index),
    )

    axes = ts.plot(index=(slice(None), slice(None), 1))
    assert len(axes) == 2
    assert reduce(lambda x, y: x + y, [len(ax.get_lines()) for ax in axes]) == 2


def test_ts_get_idx_combinations(ts_data):
    ts = TimeSeries(ts_data)
    idx_comb = ts._get_idx_combinations(ts.ts)
    assert (idx_comb == ts_data.droplevel(0).index.unique()).all()


def test_ts_plot_seasonal_default(ts_data):
    ts = TimeSeries(ts_data)

    axes = ts.plot_seasonal(
        column="sales",
        period="week",
        freq="day_of_week",
        index=(slice(None), "GROCERY I", 1),
    )

    wks = ts.ts.index.get_level_values(0).week.unique()
    assert len(axes[0].get_lines()) == len(wks)
    assert axes[0].get_lines()[0]._alpha == 1 / np.sqrt(len(wks))
    assert axes[0].get_lines()[0].get_color() == "red"
    assert axes[0].get_lines()[0].get_linewidth() == 0.5
    assert axes[0].get_xlabel() == "day_of_week"
    assert axes[0].get_ylabel() == "sales"
    assert axes[0].get_title() == "Season (week) for ('GROCERY I', 1)"


def test_ts_plot_non_mono_seasonal(ts_data):
    ts = TimeSeries(ts_data)
    axes = ts.plot_seasonal(
        column="sales", period="week", freq="day_of_week", monochrome=False
    )

    assert axes[0].get_lines()[0].get_color() == "#1f77b4"
    assert axes[0].get_lines()[1].get_color() == "#ff7f0e"
    assert axes[0].get_lines()[2].get_color() == "#2ca02c"


def test_ts_plot_seasonal_annot(ts_data):
    ts = TimeSeries(ts_data)
    axes = ts.plot_seasonal(
        column="sales",
        period="week",
        freq="day_of_week",
        index=(slice(None), "GROCERY I", 1),
        annot=True,
    )

    assert len(axes[0].texts) == 3 * 2  # 3 weeks, 2 annotations per week
    assert axes[0].texts[0].get_text() == "1"
    assert axes[0].texts[1].get_text() == "1"
    assert axes[0].texts[0].get_position() == (1, 0)
    assert axes[0].texts[1].get_position() == (6, 723)
    assert axes[0].texts[4].get_text() == "3"
    assert axes[0].texts[4].get_position() == (0, 1902)
    assert axes[0].texts[5].get_position() == (1, 1671)

    assert axes[0].texts[0].get_color() == "red"


def test_ts_plot_subseries(ts_data):
    ts = TimeSeries(ts_data)
    uniq_idx_comb = ts._get_idx_combinations(ts.ts)

    fig = ts.plot_subseries(column="sales", freq="day_of_week")

    # Check suptitle
    assert fig.get_suptitle() == "Subseries for sales by day_of_week"
    axes = fig.get_axes()
    assert len(axes) == len(uniq_idx_comb) * 7
    ax_0 = axes[0]
    assert ax_0.get_ylabel() == "sales"
    ax_1 = axes[1]
    assert ax_1.get_ylabel() == ""

    # Check line color
    assert ax_0.get_lines()[0].get_color() == "black"
    assert ax_0.get_lines()[0].get_linewidth() == 0.5
    assert ax_0.get_lines()[1].get_color() == "red"
    assert ax_0.get_lines()[1].get_linewidth() == 1
    assert ax_0.get_lines()[1].get_linestyle() == "--"

    # Check wspace == 0 of subfigures
    subfigs = [subfig for subfig in fig.get_children() if isinstance(subfig, SubFigure)]
    assert len(subfigs) == len(uniq_idx_comb)


def test_ts_plot_subseries_resampled(ts_data):
    """
    Adequate use of resampler to make sure comparisons are at a useful
    aggregation level (e.g. splitting by month but looking at daily level
    is rarely useful as there will be large flat lines in between two Januaries)
    """

    ts = TimeSeries(ts_data)
    uniq_idx_comb = ts._get_idx_combinations(ts.ts)
    idx = uniq_idx_comb[0]

    fig = ts.plot_subseries(column="sales", freq="month", resampler="ME")

    ts_data_check = ts._get_timestamp_index()
    ts_data_check = (
        ts_data_check.loc[(slice(None), *idx)]
        .groupby(pd.Grouper(freq="ME", level=0))
        .mean()
        .dropna()
    )["sales"].values

    y = fig.get_axes()[0].get_lines()[0].get_ydata()
    assert len(y) == 1
    assert (ts_data_check == y).all()


def test_ts_plot_subseries_xgranularity(ts_data):
    """
    Test that the x-axis granularity is set to the frequency of the subseries.
    """

    ts = TimeSeries(ts_data)
    fig = ts.plot_subseries(column="sales", freq="month", x_granularity="year")

    axes = fig.get_axes()
    for ax in axes:
        assert ax.get_xlabel() == "year"
        assert ax.get_lines()[0].get_xdata()[0] == 2013


def test_ts_plot_pairs(ts_corr_dummy_data):
    df = ts_corr_dummy_data
    df["add_corr_col"] = np.random.rand(len(df))
    ts = TimeSeries(ts_corr_dummy_data)

    idx = pd.IndexSlice["2014", slice(None), slice(None)]
    axes = ts.plot_pairs(index=idx)

    assert len(axes.flatten()) == len(df.columns) ** 2
    assert isinstance(axes[0][0].get_lines()[0], plt.Line2D)
    assert axes[1][0].has_data()
    assert axes[0][0].get_title() == "sales"
    assert axes[0][1].get_title() == "onpromotion"
    assert axes[0][0].get_ylabel() == "sales"
    assert axes[1][0].get_ylabel() == "onpromotion"
    assert isinstance(axes[1][0].get_children()[0], PathCollection)
    assert isinstance(axes[0][1].get_children()[0], plt.Text)
    corr_pair = ts.ts.loc[idx, ["sales", "onpromotion"]]
    r_2 = np.round(np.corrcoef(corr_pair["sales"], corr_pair["onpromotion"])[0][1], 2)
    assert axes[0][1].get_children()[0].get_text() == f"r^2 = {r_2}"
    assert axes[0][1].get_children()[0].get_text() != f"nan"

    r_2_rand = np.round(
        np.corrcoef(corr_pair["sales"], df.loc[idx, "add_corr_col"])[0][1], 2
    )
    assert axes[0][2].get_children()[0].get_text() == f"r^2 = {r_2_rand}"


def test_ts_plot_pairs_hue(ts_corr_dummy_data):
    df = ts_corr_dummy_data
    df["day_of_week"] = df.index.get_level_values(0).dayofweek.astype("str")
    ts = TimeSeries(df)

    idx = pd.IndexSlice["2014", slice(None), slice(None)]
    axes = ts.plot_pairs(index=idx, hue="day_of_week")

    # Force plot to render to check that the hue is working
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    assert isinstance(axes.flatten()[2].get_children()[0], PathCollection)
    assert (
        len(np.unique(axes.flatten()[2].get_children()[0].get_facecolors(), axis=0))
        == 7
    )
