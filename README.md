<div align="center">

# 📈 pine_ta

### TradingView's Pine Script indicators, in pure pandas and numpy

No TA-Lib. No C compiler. Just `pip install` and go.

[![PyPI](https://img.shields.io/pypi/v/pine-ta?style=flat&logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/pine-ta/)
[![Python](https://img.shields.io/pypi/pyversions/pine-ta?style=flat&logo=python&logoColor=white)](https://pypi.org/project/pine-ta/)
[![Tests](https://img.shields.io/github/actions/workflow/status/huzaifazahoor/pine_ta/publish.yml?style=flat&logo=githubactions&logoColor=white&label=tests)](https://github.com/huzaifazahoor/pine_ta/actions)
[![Downloads](https://img.shields.io/pypi/dm/pine-ta?style=flat&color=success)](https://pypi.org/project/pine-ta/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](https://github.com/huzaifazahoor/pine_ta/blob/main/LICENSE)

![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![TradingView](https://img.shields.io/badge/Pine_Script-131722?style=for-the-badge&logo=tradingview&logoColor=white)

</div>

---

## 💡 Why pine_ta?

I've spent years turning clients' TradingView indicators into Python. The same problems kept coming up: TA-Lib needs C libraries that break installs, and most Python libraries give numbers that don't match the chart. pine_ta fixes both.

- **Pine Script formulas.** Uses the same RMA, EMA, and warm-up rules as TradingView, so early bars are empty instead of fake numbers.
- **Pure Python.** Only pandas and numpy. Installs anywhere.
- **Works with your data.** `Close`, `close`, `close_price`, `c`, or yfinance's two-level columns all just work.
- **Never touches your DataFrame.** Every call returns a new result.
- **Clear errors.** Missing or text columns tell you exactly what's wrong and how to fix it.

---

## 📦 Installation

```bash
pip install pine_ta
```

Requires Python 3.10+.

---

## 🚀 Quick Start

```python
import yfinance as yf
import pine_ta

df = yf.download("AAPL", period="1y", auto_adjust=False, multi_level_index=False)

df["rsi"] = df.pine.rsi(14)
df["macd"], df["signal"], df["hist"] = df.pine.macd()
df["atr"] = df.pine.atr(14)
```

Importing `pine_ta` adds `.pine` to every DataFrame. That's it.

---

## 🧩 Column Names

pine_ta finds your columns no matter how they're named:

| Field | Works with |
|---|---|
| close | `close`, `Close`, `CLOSE`, `close_price`, `Close Price`, `c`, `last`, `price` |
| open, high, low | `Open`, `open_price`, `o`, `h`, `l`, and more |
| volume | `volume`, `Volume`, `vol`, `v`, `total_volume` |

Use a different column with an override:

```python
df.pine.rsi(14, close="Adj Close")
```

> ⚠️ `Adj Close` is never picked by default. TradingView uses the normal close, so pine_ta does too.

---

## 🛑 Helpful Errors

```python
>>> df.drop(columns=["Close"]).pine.rsi()
ColumnNotFoundError: rsi needs: close. Your DataFrame has: Adj Close, High, Low,
Open, Volume. Rename the column or pass it directly, e.g. df.pine.rsi(close='YourColumnName').

>>> df.pine.rsi()   # when Close was loaded as text
ColumnTypeError: rsi needs numbers, but 'Close' is str.
Convert it first, e.g. df['Close'] = pd.to_numeric(df['Close']).
```

Catch every library error with one line:

```python
try:
    df.pine.rsi()
except pine_ta.PineTAError as e:
    print(e)
```

---

## 📊 Indicators

| Indicator | Method | Pine Script | Needs | Returns |
|---|---|---|---|---|
| RSI | `rsi()` | `ta.rsi` | close | Series |
| MACD | `macd()` | `ta.macd` | close | macd, signal, hist |
| ADX | `adx()` | `ta.dmi` | high, low, close | Series |
| ATR | `atr()` | `ta.atr` | high, low, close | Series |
| True Range | `true_range()` | `ta.tr` | high, low, close | Series |
| Bollinger Bands | `bollinger_bands()` | `ta.bb` | close | basis, upper, lower |
| Keltner Channels | `keltner_channels()` | `ta.kc` | high, low, close | middle, upper, lower |
| CCI | `cci()` | `ta.cci` | high, low, close | Series |
| Stochastic | `stochastic()` | `ta.stoch` | high, low, close | %K, %D |
| Williams %R | `williams_percent_range()` | `ta.wpr` | high, low, close | Series |
| Rate of Change | `roc()` | `ta.roc` | close | Series |
| Momentum | `momentum()` | `ta.mom` | close | Series |
| On-Balance Volume | `obv()` | `ta.obv` | close, volume | Series |
| Money Flow Index | `mfi()` | `ta.mfi` | high, low, close, volume | Series |
| Relative Volatility Index | `rvi()` | built-in RVI | close | Series |
| Stochastic Momentum Index | `smi()` | built-in SMI | high, low, close | smi, signal |
| Awesome Oscillator | `awesome_oscillator()` | built-in AO | high, low | Series |
| MA Envelope Slope | `ma_envelope_slope()` | custom | close | float |
| Performance Index | `performance_index(benchmark_df)` | custom | date, close, symbol | Series |
| Comparative Relative Strength | `crs(benchmark_df)` | custom | date, close, symbol | Series |

See them all in code:

```python
pine_ta.list_indicators()
help(df.pine.rsi)
```

---

## 🔧 Plain Functions

Moving averages work on any Series, including other indicators:

```python
from pine_ta import ema, rma, wma, vwma, stdev

fast = ema(df["Close"], 12)
smooth_rsi = rma(df.pine.rsi(14), 5)
```

---

## ➕ Add Your Own Indicator

Write one function and register it. It gets column lookup, errors, and `df.pine` for free:

```python
import pine_ta

@pine_ta.register(("high", "low"))
def hl2(df):
    return (df["high"] + df["low"]) / 2

df.pine.hl2()  # works with "High", "LOW", "h", ...
```

---

## 🧪 Development

```bash
git clone https://github.com/huzaifazahoor/pine_ta.git
cd pine_ta
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
pytest
```

Found a value that doesn't match TradingView? [Open an issue](https://github.com/huzaifazahoor/pine_ta/issues) with the symbol, timeframe, date, and both values.

---

## 🗺️ Roadmap

- [ ] Test values against exported TradingView chart data
- [ ] More indicators: Supertrend, Ichimoku, VWAP, Parabolic SAR
- [ ] Multi-ticker support

---

## 👤 Author

**Huzaifa Zahoor** · Senior Software & Data Engineer building stock market tools

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat&logo=googlechrome&logoColor=white)](https://huzaifazahoor.online)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/huzaifazahoor)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/huzaifazahoor)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@huzaifazahoor654)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white)](https://x.com/huzaifazahoor2)

---

<div align="center">

MIT License · If pine_ta saves you time, give it a ⭐

</div>