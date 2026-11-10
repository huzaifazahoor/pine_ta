"""
The ``df.pine`` accessor.

Importing pine_ta registers ``pine`` on every pandas DataFrame, so users
can write ``df.pine.rsi(14)``. Each call:

1. Finds the columns the indicator needs (any naming style).
2. Builds an internal copy with standard names.
3. Runs the indicator on that copy.
4. Returns the result lined up with the user's index.

The user's DataFrame is never changed.
"""

import inspect

import pandas as pd

from .columns import ALIASES, prepare
from .registry import INDICATORS

# Keyword arguments that pick a column instead of an indicator setting,
# e.g. df.pine.rsi(close="Adj Close").
COLUMN_OVERRIDES = tuple(ALIASES)


@pd.api.extensions.register_dataframe_accessor("pine")
class PineAccessor:
    """
    Run pine_ta indicators directly on a DataFrame.

    Every registered indicator is available as a method, for example
    ``df.pine.rsi()``, ``df.pine.macd()``, ``df.pine.atr(14)``.

    Column overrides
    ----------------
    Every method also accepts ``open``, ``high``, ``low``, ``close``,
    ``volume``, ``date``, and ``symbol`` keyword arguments to pick a
    column by its real name, e.g. ``df.pine.rsi(close="Adj Close")``.

    Examples
    --------
    >>> import pine_ta
    >>> df["rsi"] = df.pine.rsi(14)
    >>> df["macd"], df["signal"], df["hist"] = df.pine.macd()
    """

    def __init__(self, df):
        self._df = df

    def __getattr__(self, name):
        spec = INDICATORS.get(name)
        if spec is None:
            raise AttributeError(
                f"pine_ta has no indicator '{name}'. "
                f"Available: {', '.join(sorted(INDICATORS))}."
            )
        return self._build_method(name, spec)

    def __dir__(self):
        # Lets IDEs and dir(df.pine) show every indicator.
        return sorted(set(super().__dir__()) | set(INDICATORS))

    def _build_method(self, name, spec):
        """
        Wrap one indicator so it runs on this DataFrame.

        Parameters
        ----------
        name : str
            Indicator name.
        spec : pine_ta.registry.IndicatorSpec

        Returns
        -------
        callable
        """
        func = spec.func
        signature = inspect.signature(func)

        def method(*args, **kwargs):
            overrides = {k: kwargs.pop(k) for k in COLUMN_OVERRIDES if k in kwargs}

            # Fill in df as a placeholder so we can read every argument.
            bound = signature.bind(None, *args, **kwargs)
            bound.apply_defaults()
            first = next(iter(signature.parameters))

            fields = spec.needed_fields(bound.arguments)
            bound.arguments[first] = prepare(self._df, fields, name, overrides)

            if spec.benchmark:
                benchmark = bound.arguments.get("benchmark_df")
                if isinstance(benchmark, pd.DataFrame) and not benchmark.empty:
                    bound.arguments["benchmark_df"] = prepare(
                        benchmark, ["date", "close"], f"{name} (benchmark_df)"
                    )

            result = func(*bound.args, **bound.kwargs)
            return self._align(result, name)

        method.__name__ = name
        method.__doc__ = func.__doc__
        method.__signature__ = signature
        return method

    def _align(self, result, name):
        """
        Put the user's index back on the result, and name single outputs.

        Parameters
        ----------
        result : pandas.Series, tuple of Series, or scalar
        name : str

        Returns
        -------
        Same type as ``result``.
        """
        if isinstance(result, tuple):
            return tuple(self._align(r, None) for r in result)

        if isinstance(result, pd.Series):
            if len(result) == len(self._df):
                result = result.set_axis(self._df.index)
            if name is not None:
                result = result.rename(name)

        return result
