"""
pine_ta: Pine Script style technical indicators in pure pandas and numpy.

Quick start
-----------
>>> import pine_ta
>>> df["rsi"] = df.pine.rsi(14)

You can also call indicators as plain functions:

>>> from pine_ta import ema
>>> fast = ema(df["close"], 12)
"""

from importlib.metadata import PackageNotFoundError, version

from . import indicators  # noqa: F401
from .accessor import PineAccessor
from .exceptions import ColumnNotFoundError, ColumnTypeError, PineTAError
from .indicators import *  # noqa: F401,F403
from .indicators import __all__ as _indicator_names
from .registry import INDICATORS, register

try:
    __version__ = version("pine_ta")
except PackageNotFoundError:  # running from source, not installed
    __version__ = "0.0.0"


def list_indicators():
    """
    Return the names of all indicators available on ``df.pine``.

    Returns
    -------
    list of str
    """
    return sorted(INDICATORS)


__all__ = [
    "PineAccessor",
    "PineTAError",
    "ColumnNotFoundError",
    "ColumnTypeError",
    "register",
    "list_indicators",
    "__version__",
    *_indicator_names,
]
