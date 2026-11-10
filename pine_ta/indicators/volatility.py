"""
Volatility indicators.

ATR, Bollinger Bands, Keltner Channels, and Relative Volatility Index.
"""

import numpy as np
import pandas as pd

from ..registry import register
from .core import ema, rma, stdev, true_range, vwma, wma


@register(("high", "low", "close"))
def atr(df, length=14, smoothing="RMA"):
    """
    Average True Range, like Pine Script's ``ta.atr``.

    Needs columns: high, low, close.

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Default 14.
    smoothing : {"RMA", "SMA", "EMA", "WMA"}, optional
        Default "RMA", which matches Pine Script.

    Returns
    -------
    pandas.Series
    """
    # Calculate True Range
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Apply smoothing
    if smoothing == "RMA":
        return rma(tr, length)
    elif smoothing == "SMA":
        return tr.rolling(window=length).mean()
    elif smoothing == "EMA":
        return ema(tr, length)
    elif smoothing == "WMA":
        return wma(tr, length)


def _bb_fields(args):
    fields = [args.get("src_col", "close")]
    if args.get("ma_type", "SMA") == "VWMA":
        fields.append("volume")
    return fields


@register(_bb_fields)
def bollinger_bands(df, length=20, mult=2.0, ma_type="SMA", src_col="close"):
    """
    Bollinger Bands, like Pine Script's ``ta.bb``.

    Needs columns: ``src_col`` (default close), plus volume for VWMA.

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Default 20.
    mult : float, optional
        Band width in standard deviations. Default 2.0.
    ma_type : {"SMA", "EMA", "RMA", "SMMA (RMA)", "WMA", "VWMA"}, optional
        Basis line type. Default "SMA".
    src_col : str, optional
        Default "close".

    Returns
    -------
    tuple of pandas.Series
        (basis, upper, lower)
    """
    src = df[src_col]

    # Calculate basis MA
    if ma_type == "SMA":
        basis = src.rolling(window=length).mean()
    elif ma_type == "EMA":
        basis = ema(src, length)
    elif ma_type == "SMMA (RMA)" or ma_type == "RMA":
        basis = rma(src, length)
    elif ma_type == "WMA":
        basis = wma(src, length)
    elif ma_type == "VWMA":
        basis = vwma(src, df["volume"], length)

    # Calculate bands
    dev = mult * stdev(src, length)
    upper = basis + dev
    lower = basis - dev

    return basis, upper, lower


@register(
    lambda args: list(
        dict.fromkeys([args.get("src_col", "close"), "high", "low", "close"])
    )
)
def keltner_channels(
    df,
    length=20,
    mult=2.0,
    src_col="close",
    use_exp=True,
    bands_style="Average True Range",
    atr_length=10,
):
    """
    Keltner Channels, like Pine Script's ``ta.kc``.

    Needs columns: high, low, close, and ``src_col`` (default close).

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Middle line length. Default 20.
    mult : float, optional
        Band multiplier. Default 2.0.
    src_col : str, optional
        Default "close".
    use_exp : bool, optional
        True for EMA middle line, False for SMA. Default True.
    bands_style : {"Average True Range", "True Range", "Range"}, optional
        How band width is measured.
    atr_length : int, optional
        ATR length when bands_style is "Average True Range". Default 10.

    Returns
    -------
    tuple of pandas.Series
        (middle, upper, lower)
    """
    src = df[src_col]

    # Calculate MA
    if use_exp:
        ma = ema(src, length)
    else:
        ma = src.rolling(window=length).mean()

    # Calculate range
    if bands_style == "True Range":
        rangema = true_range(df)
    elif bands_style == "Average True Range":
        rangema = atr(df, length=atr_length, smoothing="RMA")
    else:  # "Range"
        rangema = rma(df["high"] - df["low"], length)

    # Calculate bands
    upper = ma + rangema * mult
    lower = ma - rangema * mult

    return ma, upper, lower


@register(("close",))
def rvi(df, length=10, ema_length=14):
    """
    Relative Volatility Index, like TradingView's built-in RVI.

    Needs columns: close.

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Stdev length. Default 10.
    ema_length : int, optional
        Smoothing length. Default 14.

    Returns
    -------
    pandas.Series
        Values from 0 to 100.
    """
    src = df["close"]

    # Calculate standard deviation
    stddev = stdev(src, length)

    # Calculate price change
    change = src.diff()

    # Upper: EMA of stddev when price goes up (change > 0)
    upper_values = stddev.where(change > 0, 0)
    upper = ema(upper_values, ema_length)

    # Lower: EMA of stddev when price goes down (change <= 0)
    lower_values = stddev.where(change <= 0, 0)
    lower = ema(lower_values, ema_length)

    # Calculate RVI
    rvi = 100 * upper / (upper + lower)

    # Handle division by zero
    rvi = rvi.replace([float("inf"), -float("inf")], np.nan)
    rvi = rvi.fillna(50)  # When both upper and lower are 0

    return rvi
