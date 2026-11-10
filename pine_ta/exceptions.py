"""
Custom exceptions for pine_ta.

All pine_ta errors inherit from PineTAError, so users can catch
every library error with a single ``except PineTAError``.
"""


class PineTAError(Exception):
    """Base class for all pine_ta errors."""


class ColumnNotFoundError(PineTAError, KeyError):
    """
    Raised when an indicator needs a column the DataFrame does not have.

    Parameters
    ----------
    indicator : str
        Name of the indicator that was called, e.g. "rsi".
    missing : list of str
        Standard field names that could not be found, e.g. ["close"].
    available : list of str
        Column names the DataFrame actually has.
    """

    def __init__(self, indicator, missing, available):
        self.indicator = indicator
        self.missing = list(missing)
        self.available = [str(c) for c in available]
        super().__init__(self._build_message())

    def _build_message(self):
        standard = ("open", "high", "low", "close", "volume", "date", "symbol")
        first = self.missing[0]
        hint = (
            f"{first}='YourColumnName'"
            if first in standard
            else "src_col='YourColumnName'"
        )
        return (
            f"{self.indicator} needs: {', '.join(self.missing)}. "
            f"Your DataFrame has: {', '.join(self.available)}. "
            f"Rename the column or pass it directly, "
            f"e.g. df.pine.{self.indicator}({hint})."
        )

    def __str__(self):
        # KeyError wraps messages in quotes by default; show plain text instead.
        return self.args[0]


class ColumnTypeError(PineTAError, TypeError):
    """
    Raised when a needed column is not numeric.

    Parameters
    ----------
    indicator : str
        Name of the indicator that was called.
    bad_columns : dict
        Real column name to its dtype, e.g. {"Close": "object"}.
    """

    def __init__(self, indicator, bad_columns):
        self.indicator = indicator
        self.bad_columns = dict(bad_columns)
        details = ", ".join(f"'{c}' is {t}" for c, t in self.bad_columns.items())
        first = next(iter(self.bad_columns))
        super().__init__(
            f"{indicator} needs numbers, but {details}. "
            f"Convert it first, e.g. df['{first}'] = pd.to_numeric(df['{first}'])."
        )
