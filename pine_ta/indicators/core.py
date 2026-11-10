"""
Core moving averages and helpers.

Most functions here take a pandas Series, not a DataFrame, so they work
on any data, including the output of other indicators. Other indicator
modules build on these.
"""

import numpy as np
import pandas as pd

from ..registry import register


def ema(src, length):
    """
    Exponential moving average, like Pine Script's ``ta.ema``.

    Parameters
    ----------
    src : pandas.Series
        Input values.
    length : int
        Number of bars.

    Returns
    -------
    pandas.Series
    """
    alpha = 2 / (length + 1)
    ema = src.copy()

    for i in range(1, len(src)):
        if pd.isna(ema.iloc[i - 1]):
            ema.iloc[i] = src.iloc[i]
        else:
            ema.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * ema.iloc[i - 1]

    return ema


def rma(src, length):
    """
    Running moving average, like Pine Script's ``ta.rma``.

    Returns NaN until there are ``length`` valid values, starts with
    an SMA, then smooths with alpha = 1 / length.

    Parameters
    ----------
    src : pandas.Series
        Input values.
    length : int
        Number of bars.

    Returns
    -------
    pandas.Series
    """
    alpha = 1 / length
    values = src.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    prev = np.nan

    for i in range(len(values)):
        if np.isnan(prev):
            # Same as Pine: na(sum[1]) ? ta.sma(src, length) : ...
            if i >= length - 1:
                window = values[i - length + 1 : i + 1]
                if not np.isnan(window).any():
                    out[i] = window.mean()
        else:
            out[i] = alpha * values[i] + (1 - alpha) * prev
        prev = out[i]

    return pd.Series(out, index=src.index)


def wma(src, length):
    """
    Weighted moving average, like Pine Script's ``ta.wma``.

    Parameters
    ----------
    src : pandas.Series
        Input values.
    length : int
        Number of bars.

    Returns
    -------
    pandas.Series
    """
    weights = np.arange(length, 0, -1)
    wma = pd.Series(np.nan, index=src.index)

    for i in range(length - 1, len(src)):
        window = src.iloc[i - length + 1 : i + 1]
        if window.notna().all():
            wma.iloc[i] = np.sum(window.values * weights) / np.sum(weights)

    return wma


def vwma(src, volume, length):
    """
    Volume weighted moving average, like Pine Script's ``ta.vwma``.

    Parameters
    ----------
    src : pandas.Series
        Input values.
    volume : pandas.Series
        Volume for each bar.
    length : int
        Number of bars.

    Returns
    -------
    pandas.Series
    """
    return (src * volume).rolling(window=length).sum() / volume.rolling(
        window=length
    ).sum()


def stdev(src, length, biased=True):
    """
    Standard deviation, like Pine Script's ``ta.stdev``.

    Parameters
    ----------
    src : pandas.Series
        Input values.
    length : int
        Number of bars.
    biased : bool, optional
        True (default) matches Pine Script. False uses the sample version.

    Returns
    -------
    pandas.Series
    """
    if biased:
        return src.rolling(window=length).std(ddof=0)
    else:
        return src.rolling(window=length).std(ddof=1)


def mean_deviation(src, length):
    """
    Mean absolute deviation, like Pine Script's ``ta.dev``.

    Parameters
    ----------
    src : pandas.Series
        Input values.
    length : int
        Number of bars.

    Returns
    -------
    pandas.Series
    """
    return src.rolling(window=length).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )


@register(("high", "low", "close"))
def true_range(df):
    """
    True range, like Pine Script's ``ta.tr``.

    Needs columns: high, low, close.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr
