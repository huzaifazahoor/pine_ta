"""
Momentum and trend indicators.

RSI, MACD, ADX, Awesome Oscillator, CCI, SMI, Williams %R,
Stochastic, ROC, Momentum, and MA envelope slope.
"""

import numpy as np
import pandas as pd

from ..registry import register
from .core import ema, mean_deviation, rma


@register(("close",))
def rsi(df, period=14):
    """
    Relative Strength Index, like Pine Script's ``ta.rsi``.

    Needs columns: close.

    Parameters
    ----------
    df : pandas.DataFrame
    period : int, optional
        Number of bars. Default 14.

    Returns
    -------
    pandas.Series
        Values from 0 to 100.
    """
    change = df["close"].diff()
    up = rma(change.clip(lower=0), period)
    down = rma((-change).clip(lower=0), period)

    # Same rules as TradingView's built-in RSI.
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(
            down == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / down))
        )

    rsi = pd.Series(values, index=df.index)
    return rsi.where(up.notna() & down.notna())


@register(("close",))
def macd(
    df,
    fast_length=12,
    slow_length=26,
    signal_length=9,
    ma_type="EMA",
    signal_type="EMA",
):
    """
    MACD, like Pine Script's ``ta.macd``.

    Needs columns: close.

    Parameters
    ----------
    df : pandas.DataFrame
    fast_length : int, optional
        Fast MA length. Default 12.
    slow_length : int, optional
        Slow MA length. Default 26.
    signal_length : int, optional
        Signal line length. Default 9.
    ma_type : {"EMA", "SMA"}, optional
        MA type for the fast and slow lines.
    signal_type : {"EMA", "SMA"}, optional
        MA type for the signal line.

    Returns
    -------
    tuple of pandas.Series
        (macd, signal, histogram)
    """
    src = df["close"]

    if ma_type == "SMA":
        fast_ma = src.rolling(window=fast_length).mean()
        slow_ma = src.rolling(window=slow_length).mean()
    else:  # EMA
        fast_ma = ema(src, fast_length)
        slow_ma = ema(src, slow_length)

    macd = fast_ma - slow_ma

    if signal_type == "SMA":
        signal = macd.rolling(window=signal_length).mean()
    else:  # EMA
        signal = ema(macd, signal_length)

    histogram = macd - signal

    return macd, signal, histogram


@register(("high", "low", "close"))
def adx(df, dilen=14, adxlen=14):
    """
    Average Directional Index, like the ADX output of Pine Script's ``ta.dmi``.

    Needs columns: high, low, close.

    Parameters
    ----------
    df : pandas.DataFrame
    dilen : int, optional
        DI length. Default 14.
    adxlen : int, optional
        ADX smoothing length. Default 14.

    Returns
    -------
    pandas.Series
    """
    # Calculate directional movement
    high_change = df["high"].diff()
    low_change = -df["low"].diff()

    # Plus and Minus DM
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    mask_plus = (high_change > low_change) & (high_change > 0)
    mask_minus = (low_change > high_change) & (low_change > 0)

    plus_dm[mask_plus] = high_change[mask_plus]
    minus_dm[mask_minus] = low_change[mask_minus]

    # True Range
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = rma(true_range, dilen)

    # Directional Indicators
    plus_di = 100 * rma(plus_dm, dilen) / atr
    minus_di = 100 * rma(minus_dm, dilen) / atr

    # ADX
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * di_diff / di_sum.replace(0, 1)

    adx = rma(dx, adxlen)

    return adx.fillna(0)


@register(("close",))
def ma_envelope_slope(df, period=50, lookback=5):
    """
    Slope of the MA envelope median line, as a percent of the MA.

    Needs columns: close.

    Parameters
    ----------
    df : pandas.DataFrame
    period : int, optional
        SMA length. Default 50.
    lookback : int, optional
        Number of recent MA values to fit. Default 5.

    Returns
    -------
    float or None
        A single number, or None if there isn't enough data.
    """
    ma = df["close"].rolling(window=period).mean()

    if ma.notna().sum() < lookback:
        return None

    recent_ma = ma.dropna().tail(lookback)
    x = np.arange(len(recent_ma))
    y = recent_ma.values

    slope = np.polyfit(x, y, 1)[0]
    return (slope / recent_ma.mean()) * 100


@register(("high", "low"))
def awesome_oscillator(df, fast_period=5, slow_period=34):
    """
    Awesome Oscillator (AO), like TradingView's built-in AO.

    Needs columns: high, low.

    Parameters
    ----------
    df : pandas.DataFrame
    fast_period : int, optional
        Default 5.
    slow_period : int, optional
        Default 34.

    Returns
    -------
    pandas.Series
    """
    hl2 = (df["high"] + df["low"]) / 2
    ao = hl2.rolling(window=fast_period).mean() - hl2.rolling(window=slow_period).mean()
    return ao


def _cci_fields(args):
    src_type = args.get("src_type", "hlc3")
    return ["high", "low", "close"] if src_type == "hlc3" else [src_type]


@register(_cci_fields)
def cci(df, length=20, src_type="hlc3"):
    """
    Commodity Channel Index, like Pine Script's ``ta.cci``.

    Needs columns: high, low, close (for "hlc3"), or the ``src_type`` column.

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Default 20.
    src_type : str, optional
        "hlc3" (default) or a column name like "close".

    Returns
    -------
    pandas.Series
    """
    if src_type == "hlc3":
        src = (df["high"] + df["low"] + df["close"]) / 3
    else:
        src = df[src_type]

    sma = src.rolling(window=length).mean()
    mad = mean_deviation(src, length)
    cci = (src - sma) / (0.015 * mad)

    return cci


@register(("high", "low", "close"))
def smi(df, length_k=10, length_d=3, length_ema=3):
    """
    Stochastic Momentum Index, like TradingView's built-in SMI.

    Needs columns: high, low, close.

    Parameters
    ----------
    df : pandas.DataFrame
    length_k : int, optional
        %K length. Default 10.
    length_d : int, optional
        %D smoothing length. Default 3.
    length_ema : int, optional
        Signal EMA length. Default 3.

    Returns
    -------
    tuple of pandas.Series
        (smi, smi_ema)
    """
    # Calculate highest high and lowest low
    highest_high = df["high"].rolling(window=length_k).max()
    lowest_low = df["low"].rolling(window=length_k).min()

    # Calculate ranges
    highest_lowest_range = highest_high - lowest_low
    relative_range = df["close"] - (highest_high + lowest_low) / 2

    # Double EMA smoothing
    relative_ema1 = ema(relative_range, length_d)
    relative_ema2 = ema(relative_ema1, length_d)

    range_ema1 = ema(highest_lowest_range, length_d)
    range_ema2 = ema(range_ema1, length_d)

    # Calculate SMI
    smi = 200 * (relative_ema2 / range_ema2)

    # SMI-based EMA
    smi_ema = ema(smi, length_ema)

    return smi, smi_ema


@register(lambda args: ["high", "low", args.get("src_col", "close")])
def williams_percent_range(df, length=14, src_col="close"):
    """
    Williams %R, like Pine Script's ``ta.wpr``.

    Needs columns: high, low, and ``src_col`` (default close).

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Default 14.
    src_col : str, optional
        Default "close".

    Returns
    -------
    pandas.Series
        Values from -100 to 0.
    """
    src = df[src_col]
    highest = df["high"].rolling(window=length).max()
    lowest = df["low"].rolling(window=length).min()

    williams_r = 100 * (src - highest) / (highest - lowest)
    return williams_r


@register(("high", "low", "close"))
def stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3):
    """
    Stochastic Oscillator, like TradingView's built-in Stoch.

    Needs columns: high, low, close.

    Parameters
    ----------
    df : pandas.DataFrame
    fastk_period : int, optional
        Default 14.
    slowk_period : int, optional
        %K smoothing. Default 3.
    slowd_period : int, optional
        %D smoothing. Default 3.

    Returns
    -------
    tuple of pandas.Series
        (slow_k, slow_d)
    """
    lowest_low = df["low"].rolling(window=fastk_period).min()
    highest_high = df["high"].rolling(window=fastk_period).max()

    # Fast %K
    fast_k = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low))

    # Slow %K (smoothed Fast %K)
    slow_k = fast_k.rolling(window=slowk_period).mean()

    # Slow %D (smoothed Slow %K)
    slow_d = slow_k.rolling(window=slowd_period).mean()

    return slow_k, slow_d


@register(lambda args: [args.get("src_col", "close")])
def roc(df, length=14, src_col="close"):
    """
    Rate of Change, like Pine Script's ``ta.roc``.

    Needs columns: ``src_col`` (default close).

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Default 14.
    src_col : str, optional
        Default "close".

    Returns
    -------
    pandas.Series
        Percent change.
    """
    src = df[src_col]
    src_shifted = src.shift(length)

    # ROC = 100 * (current - previous) / previous
    roc = 100 * (src - src_shifted) / src_shifted

    # Handle division by zero
    roc = roc.replace([float("inf"), -float("inf")], np.nan)

    return roc


@register(lambda args: [args.get("src_col", "close")])
def momentum(df, length=10, src_col="close"):
    """
    Momentum, like Pine Script's ``ta.mom``.

    Needs columns: ``src_col`` (default close).

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Default 10.
    src_col : str, optional
        Default "close".

    Returns
    -------
    pandas.Series
    """
    src = df[src_col]

    # Momentum = current price - price n periods ago
    mom = src - src.shift(length)

    return mom
