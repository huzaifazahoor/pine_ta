"""
Volume indicators.

On-Balance Volume and Money Flow Index.
"""

import numpy as np

from ..registry import register


@register(("close", "volume"))
def obv(df):
    """
    On-Balance Volume, like Pine Script's ``ta.obv``.

    Needs columns: close, volume.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.Series
    """
    src = df["close"]
    volume = df["volume"]

    # Calculate price change
    change = src.diff()

    # Sign of change: 1 if positive, -1 if negative, 0 if no change
    sign = np.sign(change)

    # OBV = cumulative sum of (sign * volume)
    obv = (sign * volume).cumsum()

    # Handle NaN in first row
    obv = obv.fillna(0)

    return obv


@register(("high", "low", "close", "volume"))
def mfi(df, length=14):
    """
    Money Flow Index, like Pine Script's ``ta.mfi``.

    Needs columns: high, low, close, volume.

    Parameters
    ----------
    df : pandas.DataFrame
    length : int, optional
        Default 14.

    Returns
    -------
    pandas.Series
        Values from 0 to 100.
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]

    # Calculate price change
    change = hlc3.diff()

    # Positive money flow: volume * hlc3 when price goes up
    positive_mf = (volume * hlc3).where(change > 0, 0)

    # Negative money flow: volume * hlc3 when price goes down
    negative_mf = (volume * hlc3).where(change < 0, 0)

    # Sum over period
    upper = positive_mf.rolling(window=length).sum()
    lower = negative_mf.rolling(window=length).sum()

    # Calculate MFI
    mfi = 100 - (100 / (1 + upper / lower))

    # Handle division by zero
    mfi = mfi.replace([float("inf"), -float("inf")], 100)
    mfi = mfi.fillna(50)

    return mfi
