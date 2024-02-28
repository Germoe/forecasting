import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import scipy
import math


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
        self.non_date_index_levels = list(range(1, len(self.ts.index.levels)))

    def __repr__(self) -> str:
        return self.ts.head().__repr__()

    def _get_timestamp_index(self) -> None:
        """
        Convert the PeriodIndex to a TimestampIndex.
        """
        ts_dt = self.ts.copy().unstack(level=self.non_date_index_levels)
        ts_dt.index = ts_dt.index.to_timestamp()
        ts_dt = ts_dt.stack(level=self.non_date_index_levels, future_stack=True)
        return ts_dt

    def _get_idx_combinations(self, data: pd.DataFrame) -> list:
        """
        Get all the unique index combinations.
        """
        idx_comb = data.droplevel(0).index.unique()
        # Reshape a single index to a multiindex
        if not isinstance(idx_comb[0], tuple):
            idx_comb = pd.MultiIndex.from_tuples([(idx,) for idx in idx_comb])
        return idx_comb

    def make_lags(self, col: str, lags: list[int]) -> pd.DataFrame:
        """
        Create lagged columns for the time series.

        :param col: The column to create lags for.
        :param lags: The lags to create. Is a list of integers.
        :return: A DataFrame with the lagged columns.
        """

        ts = self.ts.loc[:, [col]].copy()
        ts[[f"{col}_lag_{lag}" for lag in lags]] = (
            ts[col].groupby(level=self.non_date_index_levels).shift(lags)
        )

        ts = ts.drop(columns=[col])
        return ts

    def make_leads(self, col: str, leads: list[int]) -> pd.DataFrame:
        """
        Create lead columns for the time series.

        :param col: The column to create leads for.
        :param leads: The leads to create. Is a list of integers.
        :return: A DataFrame with the lead columns.
        """

        ts = self.ts.loc[:, [col]].copy()
        leads = [-lead for lead in leads if lead > 0]
        ts[[f"{col}_lead_{-lead}" for lead in leads]] = (
            ts[col].groupby(level=self.non_date_index_levels).shift(leads)
        )
        ts = ts.drop(columns=[col])
        return ts

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
                label=str(idx),
                color="black" if data.values.shape[1] == 1 else None,
                lw=0.5,
            )

            ax.set_title(f"{','.join(data.columns)} for {idx}")
            ax.set_xlabel(data.index.names[0])
            ax.set_ylabel(",".join(data.columns))

        plt.tight_layout()

        return axes

    def plot_seasonal(
        self,
        column: str,
        period: str,
        freq: str,
        index: tuple = None,
        monochrome: bool = True,
        annot: bool = False,
        resampler: str = None,
    ) -> plt.Axes:
        """
        Plot the seasonal decomposition of the time series.

        :param col: The column to plot.
        :param period: The period to plot. Has to be a level in the index.
        :param freq: The frequency to plot. Has to be a level in the index.

        :return: A matplotlib Axes object.
        """
        ts = self._get_timestamp_index()
        ts = ts.loc[:, [column]]

        if index is not None:
            ts = ts.loc[index, :].copy()

        unique_idx_comb = self._get_idx_combinations(ts)
        nrows = len(unique_idx_comb)
        ncols = 1

        if resampler is not None:
            # Resample level 0
            ts = (
                ts.unstack(level=[1])
                .groupby(pd.Grouper(freq=resampler, level=0))
                .sum()
                .stack(level=[1], future_stack=True)
            )

        ts[period] = ts.index.get_level_values(0).map(lambda x: getattr(x, period))
        ts[freq] = ts.index.get_level_values(0).map(lambda x: getattr(x, freq))
        periods = ts.loc[:, period].unique()
        alpha = 0.8

        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 6))

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, idx in enumerate(unique_idx_comb):
            sub_ts = ts.loc[(slice(None), *idx), :].copy()
            ax = axes[i]
            for p in periods:
                ts_p = sub_ts.loc[ts[period] == p, :].copy()
                ts_p = ts_p.sort_values(by=freq)

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

    def plot_pairs(self, index=None, hue=None) -> np.ndarray:
        ts = self.ts.copy()
        if index is not None:
            ts = ts.loc[index, :].copy()

        # Filter out non-numeric columns
        ts_num = ts.select_dtypes(include=[np.number])

        n = len(ts_num.columns)
        _, axes = plt.subplots(nrows=n, ncols=n, figsize=(12, 12))

        # Plot Distributions
        for i, (row_axes, row) in enumerate(zip(axes, ts_num.columns)):
            for j, (ax, col) in enumerate(zip(row_axes, ts_num.columns)):
                if i == 0:
                    ax.set_title(col)
                if i < n - 1:
                    ax.get_xaxis().set_visible(False)
                else:
                    ax.set_xlabel(col)

                    x_range = ts_num[col].max() - ts_num[col].min()
                    ax.set_xlim(
                        ts_num[col].min() - 0.1 * x_range,
                        ts_num[col].max() + 0.1 * x_range,
                    )
                if j == 0:
                    ax.set_ylabel(row)

                    if i != j:
                        # Don't apply effect to the diagonal distribution plot
                        y_range = ts_num[row].max() - ts_num[row].min()
                        ax.set_ylim(
                            ts_num[row].min() - 0.1 * y_range,
                            ts_num[row].max() + 0.1 * y_range,
                        )
                else:
                    ax.get_yaxis().set_visible(False)
                loc = i - j
                if loc == 0:
                    # KDE Plot
                    kde_vals = ts_num[col].dropna()
                    kde = scipy.stats.gaussian_kde(kde_vals)
                    x = np.linspace(kde_vals.min(), kde_vals.max(), 1000)
                    y = kde(x)

                    ax.plot(x, y, color="black")
                elif loc > 0:
                    alpha = 1 / (np.log(len(ts_num[col]) * len(ts_num[row]) / 2))
                    if hue is not None:
                        color = ts[hue].map(
                            {val: i for i, val in enumerate(ts[hue].unique())}
                        )
                    else:
                        color = "black"
                    ax.scatter(
                        x=ts_num[col],
                        y=ts_num[row],
                        alpha=alpha,
                        c=color,
                        s=5,
                    )
                elif loc < 1:
                    r_2 = np.round(np.corrcoef(ts_num[row], ts_num[col])[0][1], 2)
                    ax.text(
                        0.5,
                        0.5,
                        f"r^2 = {r_2}",
                        fontsize=max(6, 17 - len(ts_num.columns)),
                        ha="center",
                        color="black",
                    )
        plt.subplots_adjust(wspace=0, hspace=0)

        return axes

    def plot_lags(self, col: str, lags: list[int]) -> np.ndarray:
        ts = TimeSeries(self.ts.loc[:, [col]])
        ts_lags = ts.make_lags(col, lags)
        ts = pd.concat([ts.ts, ts_lags], axis=1)
        ts = ts.dropna()

        padding = 0.05 * ts.max().max()
        ncols = 3

        if len(lags) < 3:
            ncols = len(lags)
            nrows = 1
        else:
            nrows = math.ceil(len(lags) / ncols)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), sharey=True
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])

        if len(axes.shape) == 1:
            axes = axes.reshape(-1, axes.shape[0])

        lag_axes = axes.flatten()

        for lag, ax in zip(lags, lag_axes):
            ax.scatter(
                ts[f"{col}_lag_{lag}"],
                ts[col],
                label=f"Lag {lag}",
                alpha=0.15,
                s=2,
                color="black",
            )
            r_2 = np.round(np.corrcoef(ts[f"{col}_lag_{lag}"], ts[col])[0][1], 2)
            ax.text(
                0.95,
                0.05,
                f"r^2 = {r_2}",
                fontsize=max(6, 17 - len(ts.columns)),
                ha="right",
                color="red",
                transform=ax.transAxes,
            )
            ax.legend(loc="upper left")

        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                if ax.has_data():
                    ax.plot(
                        [0, 1],
                        [0, 1],
                        transform=ax.transAxes,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                    )
                ax.set_xlim(ts.min().min() - padding, ts.max().max() + padding)
                ax.set_ylim(ts.min().min() - padding, ts.max().max() + padding)
                if j == 0:
                    ax.set_ylabel(col)
                else:
                    ax.get_yaxis().set_visible(False)
                if i == (axes.shape[0] - 1):
                    ax.set_xlabel(col)
                else:
                    ax.get_xaxis().set_visible(False)

        fig.suptitle(f"Lags for {col}")
        plt.subplots_adjust(wspace=0, hspace=0)
        return axes
