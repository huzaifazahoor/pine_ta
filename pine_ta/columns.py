"""
Column lookup for pine_ta.

Finds the right column for each standard field (open, high, low,
close, volume, date, symbol) no matter how the user named it, checks
that price and volume columns hold numbers, then builds an internal
DataFrame with standard names. The user's DataFrame is never changed.
"""

import pandas as pd

from .exceptions import ColumnNotFoundError, ColumnTypeError, PineTAError

# Cleaned aliases for each field, in priority order.
# "adjclose" is left out on purpose: TradingView uses the normal close.
# Users who want it can pass close="Adj Close".
ALIASES = {
    "open": ["open", "o", "openprice", "opening"],
    "high": ["high", "h", "highprice"],
    "low": ["low", "l", "lowprice"],
    "close": ["close", "c", "closeprice", "closing", "last", "lastprice", "price"],
    "volume": ["volume", "vol", "v", "totalvolume"],
    "date": ["date", "datetime", "time", "timestamp"],
    "symbol": ["symbol", "ticker"],
}

# Fields that are not prices or volume, so they skip the number check.
NON_NUMERIC_FIELDS = ("date", "symbol")


def clean_name(name):
    """
    Turn a column name into a simple form for matching.

    Lowercases it and removes everything that is not a letter or digit,
    so "Close Price", "close_price", and " CLOSE-PRICE " all become
    "closeprice".

    Parameters
    ----------
    name : object
        Any column label.

    Returns
    -------
    str
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def flatten_columns(df):
    """
    Flatten multi-level columns, such as yfinance output.

    Works when the data is for a single ticker. Raises an error if
    several tickers are present, since it can't know which one to use.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        The same data with single-level columns.

    Raises
    ------
    PineTAError
        If the DataFrame holds several tickers or the levels can't be read.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    known = {alias for aliases in ALIASES.values() for alias in aliases}

    for level in range(df.columns.nlevels):
        values = df.columns.get_level_values(level)
        if not any(clean_name(v) in known for v in values):
            continue

        for other in range(df.columns.nlevels):
            if other == level:
                continue
            tickers = df.columns.get_level_values(other).unique()
            if len(tickers) > 1:
                raise PineTAError(
                    f"DataFrame has data for several tickers: "
                    f"{', '.join(map(str, tickers))}. Pick one first, "
                    f"e.g. df.xs('{tickers[0]}', axis=1, level={other})."
                )

        flat = df.copy()
        flat.columns = values
        return flat

    raise PineTAError(
        "Could not read the multi-level columns. Please flatten them first."
    )


def find_column(columns, field):
    """
    Find the real column name for a field.

    Standard fields (like "close") are matched through ALIASES. Any other
    name (like "MyCol" from ``src_col="MyCol"``) must match a column
    exactly.

    Parameters
    ----------
    columns : iterable
        Column labels of the DataFrame.
    field : str
        Field name, e.g. "close" or a real column name.

    Returns
    -------
    object or None
        The matching column label, or None if nothing matches.
    """
    columns = list(columns)

    if field not in ALIASES:
        return field if field in columns else None

    # Reversed so the first matching column wins.
    cleaned = {clean_name(c): c for c in reversed(columns)}
    for alias in ALIASES[field]:
        if alias in cleaned:
            return cleaned[alias]
    return None


def is_number_column(series):
    """
    Check that a column holds numbers (int or float, not bool).

    Parameters
    ----------
    series : pandas.Series

    Returns
    -------
    bool
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def prepare(df, fields, indicator, overrides=None):
    """
    Build an internal DataFrame with standard column names.

    Parameters
    ----------
    df : pandas.DataFrame
        The user's DataFrame. It is not changed.
    fields : list of str
        Fields the indicator needs, e.g. ["high", "low", "close"].
    indicator : str
        Indicator name, used in error messages.
    overrides : dict, optional
        Field to column name, e.g. {"close": "Adj Close"}.
        Entries set to None are ignored.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with the same index and standard column names.

    Raises
    ------
    ColumnNotFoundError
        If any needed field can't be found.
    ColumnTypeError
        If a price or volume column does not hold numbers.
    PineTAError
        If the DataFrame has duplicate column names or several tickers.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    df = flatten_columns(df)

    # Index name dropped so a "date" index never clashes with a "date" column.
    out = pd.DataFrame(index=df.index.rename(None))
    real_names = {}
    missing = []

    for field in fields:
        if field in overrides:
            col = overrides[field] if overrides[field] in df.columns else None
        else:
            col = find_column(df.columns, field)

        if col is not None:
            data = df[col]
            if isinstance(data, pd.DataFrame):
                raise PineTAError(
                    f"Your DataFrame has more than one column named '{col}'. "
                    f"Please keep only one."
                )
            out[field] = data
            real_names[field] = col
        elif field == "date" and isinstance(df.index, pd.DatetimeIndex):
            # yfinance and many others keep dates in the index.
            out[field] = df.index
            real_names[field] = "index"
        else:
            missing.append(field)

    if missing:
        raise ColumnNotFoundError(indicator, missing, df.columns)

    # Price and volume must be numbers.
    bad = {
        real_names[field]: str(out[field].dtype)
        for field in fields
        if field not in NON_NUMERIC_FIELDS and not is_number_column(out[field])
    }
    if bad:
        raise ColumnTypeError(indicator, bad)

    return out
