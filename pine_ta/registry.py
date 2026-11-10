"""
Indicator registry for pine_ta.

Every indicator that takes a DataFrame registers itself here with the
``@register`` decorator. The ``df.pine`` accessor reads this registry,
so adding a new indicator only means writing one decorated function.
"""

from dataclasses import dataclass
from typing import Callable, Union

INDICATORS = {}


@dataclass(frozen=True)
class IndicatorSpec:
    """
    Information about one registered indicator.

    Parameters
    ----------
    func : callable
        The indicator function. Its first argument must be a DataFrame.
    fields : tuple of str or callable
        Standard columns the indicator needs, e.g. ("high", "low", "close").
        Can also be a function that takes the call's arguments (as a dict)
        and returns the list, for indicators where it depends on settings
        like ``src_col``.
    benchmark : bool
        True if the indicator also takes a ``benchmark_df`` argument.
    """

    func: Callable
    fields: Union[tuple, Callable]
    benchmark: bool = False

    def needed_fields(self, arguments):
        """
        Return the columns needed for one specific call.

        Parameters
        ----------
        arguments : dict
            The call's arguments, with defaults filled in.

        Returns
        -------
        list of str
        """
        if callable(self.fields):
            return list(self.fields(arguments))
        return list(self.fields)


def register(fields, benchmark=False):
    """
    Decorator that adds an indicator to the registry.

    Parameters
    ----------
    fields : tuple of str or callable
        See ``IndicatorSpec.fields``.
    benchmark : bool, optional
        See ``IndicatorSpec.benchmark``.

    Returns
    -------
    callable
        The same function, unchanged.

    Examples
    --------
    >>> @register(("close",))
    ... def my_indicator(df, length=10):
    ...     return df["close"].rolling(length).mean()
    """

    def decorator(func):
        if func.__name__ in INDICATORS:
            raise ValueError(f"Indicator '{func.__name__}' is already registered.")
        INDICATORS[func.__name__] = IndicatorSpec(func, fields, benchmark)
        return func

    return decorator
