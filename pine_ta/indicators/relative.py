"""
Relative strength indicators that compare a stock to a benchmark.

Performance Index and Comparative Relative Strength.
"""

import numpy as np
import pandas as pd

from ..registry import register


@register(("date", "close", "symbol"), benchmark=True)
def performance_index(df, benchmark_df, period=21):
    """
    Performance Index against a benchmark (for example the S&P 500).

    PI = (Stock Price / Benchmark Price) * (Benchmark MA / Stock MA)

    Needs columns: date, close, symbol. The benchmark needs date, close.

    Parameters
    ----------
    df : pandas.DataFrame
        Stock data.
    benchmark_df : pandas.DataFrame
        Benchmark data.
    period : int, optional
        MA length. Default 21.

    Returns
    -------
    pandas.Series
    """
    if benchmark_df.empty:
        return pd.Series(np.nan, index=df.index)
    # Ensure date columns are datetime
    df = df.copy()
    benchmark_df = benchmark_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])

    # Merge on date
    merged = pd.merge(
        df[["date", "close", "symbol"]],
        benchmark_df[["date", "close"]],
        on="date",
        suffixes=("", "_benchmark"),
        how="left",
    )

    # Calculate moving averages
    merged["stock_ma"] = merged["close"].rolling(window=period).mean()
    merged["benchmark_ma"] = merged["close_benchmark"].rolling(window=period).mean()

    # Calculate Performance Index
    merged["pi"] = (merged["close"] / merged["close_benchmark"]) * (
        merged["benchmark_ma"] / merged["stock_ma"]
    )

    # Handle division by zero
    merged["pi"] = merged["pi"].replace([float("inf"), -float("inf")], np.nan)

    return merged["pi"]


@register(("date", "close", "symbol"), benchmark=True)
def crs(df, benchmark_df, period=21):
    """
    Comparative Relative Strength against a benchmark.

    CRS = Stock Price / Benchmark Price

    Needs columns: date, close, symbol. The benchmark needs date, close.

    Parameters
    ----------
    df : pandas.DataFrame
        Stock data.
    benchmark_df : pandas.DataFrame
        Benchmark data.
    period : int, optional
        Not used yet. Kept for a consistent API.

    Returns
    -------
    pandas.Series
    """
    if benchmark_df.empty:
        return pd.Series(np.nan, index=df.index)

    # Ensure date columns are datetime
    df = df.copy()
    benchmark_df = benchmark_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])

    # Merge on date
    merged = pd.merge(
        df[["date", "close", "symbol"]],
        benchmark_df[["date", "close"]],
        on="date",
        suffixes=("", "_benchmark"),
        how="left",
    )

    # Calculate CRS
    crs = merged["close"] / merged["close_benchmark"]

    # Handle division by zero
    crs = crs.replace([float("inf"), -float("inf")], np.nan)

    return crs
