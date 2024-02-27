import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import scipy


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

    def _get_idx_combinations(self, data: pd.DataFrame) -> list:
        """
        Get all the unique index combinations.
        """
        idx_comb = data.droplevel(0).index.unique()
        return idx_comb

    def plot(self, index: tuple = None, columns: list = None) -> plt.Axes:
        """
        Plot the time series.
        """
        plot_data = self._get_timestamp_index()

        if index is not None:
            plot_data = plot_data.loc[index, :]

        if columns is not None:
            plot_data = plot_data[columns]

        uniq_idx_comb = self._get_idx_combinations(plot_data)
        nrows = len(uniq_idx_comb)
        ncols = 1

        if nrows * ncols > 50:
            raise ValueError(
                "The number of index combinations is too large to plot individually."
            )

        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 3))

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, idx in enumerate(uniq_idx_comb):
            ax = axes[i]
            data = plot_data.loc[(slice(None), *idx), :]

            ax.plot(
                data.index.get_level_values(0),
                data.values,
                label=idx,
                color="black",
                lw=0.5,
            )

        return axes

    def plot_seasonal(
        self,
        column: str,
        period: str,
        freq: str,
        index: tuple = None,
        monochrome: bool = True,
        annot: bool = False,
    ) -> plt.Axes:
        """
        Plot the seasonal decomposition of the time series.

        :param col: The column to plot.
        :param period: The period to plot. Has to be a level in the index.
        :param freq: The frequency to plot. Has to be a level in the index.

        :return: A matplotlib Axes object.
        """
        ts = self._get_timestamp_index()

        if index is not None:
            ts = ts.loc[index, :].copy()

        unique_idx_comb = self._get_idx_combinations(ts)
        nrows = len(unique_idx_comb)
        ncols = 1

        ts[period] = ts.index.get_level_values(0).map(lambda x: getattr(x, period))
        ts[freq] = ts.index.get_level_values(0).map(lambda x: getattr(x, freq))
        periods = ts.loc[:, period].unique()
        alpha = max(0.01, 1 / np.sqrt(len(periods)))

        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 3))

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, idx in enumerate(unique_idx_comb):
            sub_ts = ts.loc[(slice(None), *idx), :].copy()
            ax = axes[i]
            for p in periods:
                ts_p = sub_ts.loc[ts[period] == p, :].sort_values(by=freq).copy()
                x = ts_p[freq]
                y = ts_p[column]
                if monochrome is True:
                    ax.plot(x, y, label=p, lw=0.5, alpha=alpha, color="red")
                else:
                    ax.plot(x, y, label=p, lw=0.5, alpha=alpha)

                if annot is True:
                    # Add Period Label to start and end of line in the color of the line
                    ax.text(
                        x.iloc[0],
                        y.iloc[0],
                        p,
                        fontsize=8,
                        ha="right",
                        color=ax.get_lines()[-1].get_color(),
                    )
                    ax.text(
                        x.iloc[-1],
                        y.iloc[-1],
                        p,
                        fontsize=8,
                        ha="left",
                        color=ax.get_lines()[-1].get_color(),
                    )

            ax.set_xlabel(freq)
            ax.set_ylabel(column)
            ax.set_title(f"Season ({period}) for {idx}")

        plt.tight_layout()
        return axes

    def plot_subseries(
        self, column: str, freq: str, resampler: str = None, x_granularity=None
    ) -> plt.Figure:
        ts = self._get_timestamp_index()
        ts[freq] = ts.index.get_level_values(0).map(lambda x: getattr(x, freq))
        ts = ts.sort_values(by=freq)

        unique_idx_comb = self._get_idx_combinations(ts)
        unique_freq = ts[freq].unique()

        nrows = len(unique_idx_comb)
        ncols = len(unique_freq)

        fig = plt.figure(constrained_layout=False, figsize=(12, nrows * 6))

        # Make sure that super title is above all plots
        fig.suptitle(f"Subseries for {column} by {freq}", fontsize=16, y=1.01)

        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=nrows, ncols=1)

        if not isinstance(subfigs, np.ndarray):
            subfigs = np.array([subfigs])

        for subfig, idx in zip(subfigs, unique_idx_comb):
            sub_ts = ts.loc[(slice(None), *idx), :].copy()
            # create 1x3 subplots per subfig
            axes = subfig.subplots(nrows=1, ncols=ncols, sharey=True, sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            for i, (ax, f) in enumerate(zip(axes, unique_freq)):
                if i == 0:
                    ax.set_ylabel(column)
                else:
                    ax.get_yaxis().set_visible(False)
                ts_p = sub_ts.loc[ts[freq] == f, :].sort_index().copy()
                if resampler is not None:
                    # Resample level 0
                    ts_p = (
                        ts_p.groupby(pd.Grouper(freq=resampler, level=0))
                        .mean()
                        .dropna()
                    )

                x = ts_p.index.get_level_values(0)
                if x_granularity is not None:
                    x = x.map(lambda x: getattr(x, x_granularity))
                y = ts_p[column]
                ax.plot(x, y, label=f, lw=0.5, color="black")
                ax.axhline(y.mean(), color="red", linestyle="--", lw=1)

                ax.set_xlabel(sub_ts.index.names[0])
                if x_granularity is not None:
                    ax.set_xlabel(x_granularity)
                ax.set_title(f"{f}")

            subfig.autofmt_xdate()
            subfig.subplots_adjust(wspace=0, hspace=0)
            subfig.suptitle(f"{idx}")

        return fig

    def plot_pairs(self, index=None):
        ts = self.ts.copy()
        if index is not None:
            ts = ts.loc[index, :].copy()

        n = len(ts.columns)
        _, axes = plt.subplots(nrows=n, ncols=n, figsize=(12, 12))

        # Plot Distributions
        for i, (row_axes, row) in enumerate(zip(axes, ts.columns)):
            for j, (ax, col) in enumerate(zip(row_axes, ts.columns)):
                if i == 0:
                    ax.set_title(col)
                if i < n - 1:
                    ax.get_xaxis().set_visible(False)
                else:
                    ax.set_xlabel(col)

                    x_range = ts[col].max() - ts[col].min()
                    ax.set_xlim(
                        ts[col].min() - 0.1 * x_range, ts[col].max() + 0.1 * x_range
                    )
                if j == 0:
                    ax.set_ylabel(row)

                    if i != j:
                        # Don't apply effect to the diagonal distribution plot
                        y_range = ts[row].max() - ts[row].min()
                        ax.set_ylim(
                            ts[row].min() - 0.1 * y_range, ts[row].max() + 0.1 * y_range
                        )
                else:
                    ax.get_yaxis().set_visible(False)
                loc = i - j
                if loc == 0:
                    # KDE Plot
                    kde_vals = ts[col].dropna()
                    kde = scipy.stats.gaussian_kde(kde_vals)
                    x = np.linspace(kde_vals.min(), kde_vals.max(), 1000)
                    y = kde(x)

                    ax.plot(x, y, color="black")
                elif loc > 0:
                    alpha = 1 / (np.log(len(ts[col]) * len(ts[row]) / 2))
                    ax.scatter(
                        x=ts[col],
                        y=ts[row],
                        alpha=alpha,
                        color="black",
                        s=5,
                    )
                elif loc < 1:
                    r_2 = np.round(np.corrcoef(ts[row], ts[col])[0][1], 2)
                    ax.text(
                        0.5,
                        0.5,
                        f"r^2 = {r_2}",
                        fontsize=max(6, 17 - len(ts.columns)),
                        ha="center",
                        color="black",
                    )
        plt.subplots_adjust(wspace=0, hspace=0)

        return axes
