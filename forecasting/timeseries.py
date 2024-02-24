import pandas as pd
import matplotlib.pyplot as plt


class TimeSeries:
    """
    A class that represents a time series and adds a collection of useful visualization
    and analysis methods.

    This class only accepts pd.MultiIndex DataFrames. If a timeseries is not a MultiIndex
    please add a level 1 index with a unique identifier for the time series.

    :param df: A DataFrame with a PeriodIndex at level 0.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df.index.get_level_values(0), pd.PeriodIndex):
            raise IndexError("The DataFrame must have a PeriodIndex at level 0.")
        if df.empty:
            raise ValueError("Empty DataFrame is not allowed.")
        self.ts = df.copy().sort_index()

    def __repr__(self) -> str:
        return self.ts.head().__repr__()

    def _get_timestamp_index(self) -> None:
        """
        Convert the PeriodIndex to a TimestampIndex.
        """
        levels = len(self.ts.index.levels)
        levels = list(range(1, levels))
        ts_dt = self.ts.copy().unstack(level=levels)
        ts_dt.index = ts_dt.index.to_timestamp()
        ts_dt = ts_dt.stack(level=levels, future_stack=True)
        return ts_dt

    def plot(self, index: tuple = None, columns: list = None) -> plt.Axes:
        """
        Plot the time series.
        """
        plot_data = self._get_timestamp_index()

        if index is not None:
            plot_data = plot_data.loc[index, :]

        if columns is not None:
            plot_data = plot_data[columns]

        uniq_idx_comb = plot_data.droplevel(0).index.unique()
        nrows = len(uniq_idx_comb)
        ncols = 1

        if nrows * ncols > 50:
            raise ValueError(
                "The number of index combinations is too large to plot individually."
            )
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 3))

        for i, idx in enumerate(uniq_idx_comb):
            if nrows == 1:
                ax = axes
            else:
                ax = axes[i]
            data = plot_data.loc[(slice(None), *idx), :]
            ax.plot(data.index.get_level_values(0), data.values, label=idx)

        return axes
