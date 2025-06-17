# pine_ta

**pine_ta** is a pure `pandas` and `numpy` implementation of popular technical indicators, inspired by Pine Script. No C extensions. No TA-Lib. Built for flexibility and easy integration with any DataFrame.

## Features

- RSI, MACD, ADX, ATR, Bollinger Bands
- Stochastic, CCI, MFI, OBV, ROC, and more
- Matching Pine Script behavior (RMA, EMA, WMA)
- Single class API: `TechnicalIndicators`

## Installation

```
pip install pine_ta
```

*(coming soon on PyPI)*

## Example Usage

```python
import pandas as pd
from pine_ta import TechnicalIndicators

# Assume df is a DataFrame with 'open', 'high', 'low', 'close', 'volume'
df['rsi'] = TechnicalIndicators.rsi(df)
df['macd'], df['signal'], df['hist'] = TechnicalIndicators.macd(df)
df['adx'] = TechnicalIndicators.adx(df)
```

## Requirements

- pandas
- numpy

## License

MIT
