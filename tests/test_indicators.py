"""
Tests for pine_ta.

Run with: pytest

These tests use made-up price data, so they run offline and fast.
They check column lookup, errors, the df.pine accessor, and that
key indicators follow Pine Script's rules.
"""

import numpy as np
import pandas as pd
import pytest

import pine_ta
from pine_ta import ColumnNotFoundError, ColumnTypeError, PineTAError
from pine_ta.registry import INDICATORS

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def ohlcv():
    """100 days of made-up OHLCV data with standard lowercase names."""
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1, 100).cumsum()
    index = pd.date_range("2024-01-01", periods=100, name="date")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.3, 100),
            "high": close + rng.uniform(0.5, 1.5, 100),
            "low": close - rng.uniform(0.5, 1.5, 100),
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, 100),
        },
        index=index,
    )


# ---------------------------------------------------------- column lookup


@pytest.mark.parametrize(
    "names",
    [
        ["Open", "High", "Low", "Close", "Volume"],
        ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"],
        ["open_price", "high_price", "low_price", "close_price", "vol"],
        ["o", "h", "l", "c", "v"],
        [" Open ", "High-", "Low", "Last", "Total Volume"],
    ],
)
def test_column_name_styles(ohlcv, names):
    expected = ohlcv.pine.atr()
    renamed = ohlcv.copy()
    renamed.columns = names
    pd.testing.assert_series_equal(renamed.pine.atr(), expected)


def test_yfinance_style_multiindex(ohlcv):
    multi = ohlcv.copy()
    multi.columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    pd.testing.assert_series_equal(multi.pine.rsi(), ohlcv.pine.rsi())


def test_several_tickers_raises(ohlcv):
    multi = pd.concat({"AAPL": ohlcv, "MSFT": ohlcv}, axis=1)
    with pytest.raises(PineTAError, match="several tickers"):
        multi.pine.rsi()


def test_adj_close_is_not_used_by_default(ohlcv):
    df = ohlcv.rename(columns={"close": "Adj Close"})
    with pytest.raises(ColumnNotFoundError):
        df.pine.rsi()


def test_column_override(ohlcv):
    df = ohlcv.rename(columns={"close": "MyPrice"})
    pd.testing.assert_series_equal(df.pine.rsi(close="MyPrice"), ohlcv.pine.rsi())


def test_custom_src_col(ohlcv):
    df = ohlcv.copy()
    df["hl2"] = (df["high"] + df["low"]) / 2
    result = df.pine.roc(src_col="hl2")
    assert result.notna().sum() > 0


# ------------------------------------------------------------------ errors


def test_missing_column_message(ohlcv):
    df = ohlcv.drop(columns=["close"])
    with pytest.raises(ColumnNotFoundError) as err:
        df.pine.rsi()
    message = str(err.value)
    assert "rsi needs: close" in message
    assert "open, high, low, volume" in message


def test_missing_column_is_also_keyerror(ohlcv):
    with pytest.raises(KeyError):
        ohlcv.drop(columns=["close"]).pine.rsi()


def test_text_column_raises(ohlcv):
    df = ohlcv.copy()
    df["close"] = df["close"].astype(str)
    with pytest.raises(ColumnTypeError, match="'close'"):
        df.pine.rsi()


def test_bool_column_raises(ohlcv):
    df = ohlcv.copy()
    df["volume"] = df["volume"] > 500_000
    with pytest.raises(ColumnTypeError):
        df.pine.obv()


def test_duplicate_columns_raise(ohlcv):
    df = pd.concat([ohlcv, ohlcv[["close"]]], axis=1)
    with pytest.raises(PineTAError, match="more than one column"):
        df.pine.rsi()


def test_unknown_indicator(ohlcv):
    with pytest.raises(AttributeError, match="no indicator"):
        ohlcv.pine.not_a_real_indicator()


def test_bad_argument(ohlcv):
    with pytest.raises(TypeError):
        ohlcv.pine.rsi(not_a_setting=5)


# ---------------------------------------------------------------- accessor


def test_user_dataframe_is_not_changed(ohlcv):
    before = ohlcv.copy()
    ohlcv.pine.macd()
    pd.testing.assert_frame_equal(ohlcv, before)


def test_result_index_matches(ohlcv):
    for series in ohlcv.pine.bollinger_bands():
        assert series.index.equals(ohlcv.index)


def test_benchmark_result_lines_up(ohlcv):
    df = ohlcv.copy()
    df["symbol"] = "AAPL"
    benchmark = ohlcv[["close"]] * 1.1
    result = df.pine.performance_index(benchmark)
    assert result.index.equals(df.index)


def test_every_indicator_runs(ohlcv):
    df = ohlcv.copy()
    df["symbol"] = "AAPL"
    benchmark = ohlcv[["close"]] * 1.1

    for name, spec in INDICATORS.items():
        method = getattr(df.pine, name)
        result = method(benchmark) if spec.benchmark else method()
        assert result is not None, name


def test_plain_functions_still_work(ohlcv):
    fast = pine_ta.ema(ohlcv["close"], 12)
    assert len(fast) == len(ohlcv)


def test_list_indicators():
    names = pine_ta.list_indicators()
    assert "rsi" in names
    assert names == sorted(names)


def test_register_custom_indicator(ohlcv):
    @pine_ta.register(("close",))
    def _test_double_close(df):
        return df["close"] * 2

    try:
        result = ohlcv.rename(columns={"close": "Close"}).pine._test_double_close()
        pd.testing.assert_series_equal(
            result, (ohlcv["close"] * 2).rename("_test_double_close")
        )
    finally:
        INDICATORS.pop("_test_double_close", None)


# --------------------------------------------------- Pine Script behavior


def test_rma_warmup_and_formula():
    src = pd.Series([1.0, 2, 3, 4, 5, 6])
    result = pine_ta.rma(src, 3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)  # SMA of 1, 2, 3
    assert result.iloc[3] == pytest.approx(
        4 / 3 + 2 * 2 / 3
    )  # alpha * src + (1 - alpha) * prev


def test_rsi_warmup_is_empty(ohlcv):
    result = ohlcv.pine.rsi(14)
    assert result.iloc[:14].isna().all()
    assert result.iloc[14:].notna().all()


def test_rsi_range(ohlcv):
    result = ohlcv.pine.rsi().dropna()
    assert result.between(0, 100).all()


def test_rsi_only_up_is_100():
    df = pd.DataFrame({"close": np.arange(1.0, 31.0)})
    assert df.pine.rsi().iloc[-1] == 100


def test_rsi_only_down_is_0():
    df = pd.DataFrame({"close": np.arange(30.0, 0.0, -1)})
    assert df.pine.rsi().iloc[-1] == 0


def test_rsi_flat_is_100():
    # TradingView checks "down == 0" first, so flat prices give 100.
    df = pd.DataFrame({"close": [5.0] * 30})
    assert df.pine.rsi().iloc[-1] == 100


def test_macd_parts_add_up(ohlcv):
    macd, signal, hist = ohlcv.pine.macd()
    pd.testing.assert_series_equal(hist, macd - signal, check_names=False)


def test_bollinger_band_width(ohlcv):
    basis, upper, lower = ohlcv.pine.bollinger_bands(length=20, mult=2.0)
    expected = 2.0 * ohlcv["close"].rolling(20).std(ddof=0)
    pd.testing.assert_series_equal(upper - basis, expected, check_names=False)
