import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculate technical indicators without external dependencies."""

    @staticmethod
    def rsi(df, period=14):
        """Calculate RSI (Relative Strength Index) for a DataFrame."""
        # Calculate price changes
        delta = df["close"].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate RMA (Running Moving Average) using EWM
        # Pine Script's RMA is equivalent to EWM with alpha = 1/period
        avg_gains = gains.ewm(alpha=1 / period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1 / period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.fillna(0)  # When avg_losses is 0, RSI should be 100
        rsi = rsi.replace([float("inf"), -float("inf")], 100)

        return rsi

    @staticmethod
    def ema(src, length):
        """Calculate EMA using Pine Script's formula."""
        alpha = 2 / (length + 1)
        ema = src.copy()

        for i in range(1, len(src)):
            if pd.isna(ema.iloc[i - 1]):
                ema.iloc[i] = src.iloc[i]
            else:
                ema.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * ema.iloc[i - 1]

        return ema

    @staticmethod
    def macd(
        df,
        fast_length=12,
        slow_length=26,
        signal_length=9,
        ma_type="EMA",
        signal_type="EMA",
    ):
        """Calculate MACD with correct EMA implementation."""
        src = df["close"]

        if ma_type == "SMA":
            fast_ma = src.rolling(window=fast_length).mean()
            slow_ma = src.rolling(window=slow_length).mean()
        else:  # EMA
            fast_ma = TechnicalIndicators.ema(src, fast_length)
            slow_ma = TechnicalIndicators.ema(src, slow_length)

        macd = fast_ma - slow_ma

        if signal_type == "SMA":
            signal = macd.rolling(window=signal_length).mean()
        else:  # EMA
            signal = TechnicalIndicators.ema(macd, signal_length)

        histogram = macd - signal

        return macd, signal, histogram

    @staticmethod
    def rma(src, length):
        """Calculate RMA (Running Moving Average) used in RSI and ADX."""
        alpha = 1 / length
        rma = src.copy()

        # First non-NaN value uses SMA
        first_valid_idx = src.first_valid_index()
        if first_valid_idx is not None:
            # Convert to integer position
            first_valid_pos = src.index.get_loc(first_valid_idx)
            end_pos = min(first_valid_pos + length, len(src))

            # Calculate initial SMA
            rma.iloc[first_valid_pos:end_pos] = (
                src.iloc[first_valid_pos:end_pos]
                .rolling(window=length, min_periods=1)
                .mean()
            )

            # Apply RMA formula for subsequent values
            for i in range(end_pos, len(src)):
                if pd.notna(rma.iloc[i - 1]) and pd.notna(src.iloc[i]):
                    rma.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * rma.iloc[i - 1]

        return rma

    @staticmethod
    def adx(df, dilen=14, adxlen=14):
        """Calculate ADX (Average Directional Index)."""
        # Calculate directional movement
        high_change = df["high"].diff()
        low_change = -df["low"].diff()

        # Plus and Minus DM
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        mask_plus = (high_change > low_change) & (high_change > 0)
        mask_minus = (low_change > high_change) & (low_change > 0)

        plus_dm[mask_plus] = high_change[mask_plus]
        minus_dm[mask_minus] = low_change[mask_minus]

        # True Range
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = TechnicalIndicators.rma(true_range, dilen)

        # Directional Indicators
        plus_di = 100 * TechnicalIndicators.rma(plus_dm, dilen) / atr
        minus_di = 100 * TechnicalIndicators.rma(minus_dm, dilen) / atr

        # ADX
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = 100 * di_diff / di_sum.replace(0, 1)

        adx = TechnicalIndicators.rma(dx, adxlen)

        return adx.fillna(0)

    @staticmethod
    def ma_envelope_slope(df, period=50, lookback=5):
        """Calculate the slope of the MA envelope median line."""
        ma = df["close"].rolling(window=period).mean()

        if ma.notna().sum() < lookback:
            return None

        recent_ma = ma.dropna().tail(lookback)
        x = np.arange(len(recent_ma))
        y = recent_ma.values

        slope = np.polyfit(x, y, 1)[0]
        return (slope / recent_ma.mean()) * 100

    @staticmethod
    def awesome_oscillator(df, fast_period=5, slow_period=34):
        """Calculate Awesome Oscillator (AO)."""
        hl2 = (df["high"] + df["low"]) / 2
        ao = (
            hl2.rolling(window=fast_period).mean()
            - hl2.rolling(window=slow_period).mean()
        )
        return ao

    @staticmethod
    def wma(src, length):
        """Calculate WMA (Weighted Moving Average)."""
        weights = np.arange(length, 0, -1)
        wma = pd.Series(np.nan, index=src.index)

        for i in range(length - 1, len(src)):
            window = src.iloc[i - length + 1 : i + 1]
            if window.notna().all():
                wma.iloc[i] = np.sum(window.values * weights) / np.sum(weights)

        return wma

    @staticmethod
    def atr(df, length=14, smoothing="RMA"):
        """Calculate ATR (Average True Range)."""
        # Calculate True Range
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Apply smoothing
        if smoothing == "RMA":
            return TechnicalIndicators.rma(tr, length)
        elif smoothing == "SMA":
            return tr.rolling(window=length).mean()
        elif smoothing == "EMA":
            return TechnicalIndicators.ema(tr, length)
        elif smoothing == "WMA":
            return TechnicalIndicators.wma(tr, length)

    @staticmethod
    def vwma(src, volume, length):
        """Calculate VWMA (Volume Weighted Moving Average)."""
        return (src * volume).rolling(window=length).sum() / volume.rolling(
            window=length
        ).sum()

    @staticmethod
    def stdev(src, length, biased=True):
        """Calculate standard deviation."""
        if biased:
            return src.rolling(window=length).std(ddof=0)
        else:
            return src.rolling(window=length).std(ddof=1)

    @staticmethod
    def bollinger_bands(df, length=20, mult=2.0, ma_type="SMA", src_col="close"):
        """Calculate Bollinger Bands."""
        src = df[src_col]

        # Calculate basis MA
        if ma_type == "SMA":
            basis = src.rolling(window=length).mean()
        elif ma_type == "EMA":
            basis = TechnicalIndicators.ema(src, length)
        elif ma_type == "SMMA (RMA)" or ma_type == "RMA":
            basis = TechnicalIndicators.rma(src, length)
        elif ma_type == "WMA":
            basis = TechnicalIndicators.wma(src, length)
        elif ma_type == "VWMA":
            basis = TechnicalIndicators.vwma(src, df["volume"], length)

        # Calculate bands
        dev = mult * TechnicalIndicators.stdev(src, length)
        upper = basis + dev
        lower = basis - dev

        return basis, upper, lower

    @staticmethod
    def keltner_channels(
        df,
        length=20,
        mult=2.0,
        src_col="close",
        use_exp=True,
        bands_style="Average True Range",
        atr_length=10,
    ):
        """Calculate Keltner Channels."""
        src = df[src_col]

        # Calculate MA
        if use_exp:
            ma = TechnicalIndicators.ema(src, length)
        else:
            ma = src.rolling(window=length).mean()

        # Calculate range
        if bands_style == "True Range":
            rangema = TechnicalIndicators.true_range(df)
        elif bands_style == "Average True Range":
            rangema = TechnicalIndicators.atr(df, length=atr_length, smoothing="RMA")
        else:  # "Range"
            rangema = TechnicalIndicators.rma(df["high"] - df["low"], length)

        # Calculate bands
        upper = ma + rangema * mult
        lower = ma - rangema * mult

        return ma, upper, lower

    @staticmethod
    def true_range(df):
        """Calculate True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr

    @staticmethod
    def mean_deviation(src, length):
        """Calculate mean deviation (ta.dev in Pine Script)."""
        return src.rolling(window=length).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )

    @staticmethod
    def cci(df, length=20, src_type="hlc3"):
        """Calculate Commodity Channel Index."""
        if src_type == "hlc3":
            src = (df["high"] + df["low"] + df["close"]) / 3
        else:
            src = df[src_type]

        sma = src.rolling(window=length).mean()
        mad = TechnicalIndicators.mean_deviation(src, length)
        cci = (src - sma) / (0.015 * mad)

        return cci

    @staticmethod
    def smi(df, length_k=10, length_d=3, length_ema=3):
        """Calculate Stochastic Momentum Index."""
        # Calculate highest high and lowest low
        highest_high = df["high"].rolling(window=length_k).max()
        lowest_low = df["low"].rolling(window=length_k).min()

        # Calculate ranges
        highest_lowest_range = highest_high - lowest_low
        relative_range = df["close"] - (highest_high + lowest_low) / 2

        # Double EMA smoothing
        relative_ema1 = TechnicalIndicators.ema(relative_range, length_d)
        relative_ema2 = TechnicalIndicators.ema(relative_ema1, length_d)

        range_ema1 = TechnicalIndicators.ema(highest_lowest_range, length_d)
        range_ema2 = TechnicalIndicators.ema(range_ema1, length_d)

        # Calculate SMI
        smi = 200 * (relative_ema2 / range_ema2)

        # SMI-based EMA
        smi_ema = TechnicalIndicators.ema(smi, length_ema)

        return smi, smi_ema

    @staticmethod
    def williams_percent_range(df, length=14, src_col="close"):
        """Calculate Williams %R."""
        src = df[src_col]
        highest = df["high"].rolling(window=length).max()
        lowest = df["low"].rolling(window=length).min()

        williams_r = 100 * (src - highest) / (highest - lowest)
        return williams_r

    @staticmethod
    def stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3):
        """Calculate Stochastic Oscillator (%K and %D)."""
        lowest_low = df["low"].rolling(window=fastk_period).min()
        highest_high = df["high"].rolling(window=fastk_period).max()

        # Fast %K
        fast_k = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low))

        # Slow %K (smoothed Fast %K)
        slow_k = fast_k.rolling(window=slowk_period).mean()

        # Slow %D (smoothed Slow %K)
        slow_d = slow_k.rolling(window=slowd_period).mean()

        return slow_k, slow_d

    @staticmethod
    def roc(df, length=14, src_col="close"):
        """Calculate Rate of Change (ROC)."""
        src = df[src_col]
        src_shifted = src.shift(length)

        # ROC = 100 * (current - previous) / previous
        roc = 100 * (src - src_shifted) / src_shifted

        # Handle division by zero
        roc = roc.replace([float("inf"), -float("inf")], np.nan)

        return roc

    @staticmethod
    def momentum(df, length=10, src_col="close"):
        """Calculate Momentum indicator."""
        src = df[src_col]

        # Momentum = current price - price n periods ago
        mom = src - src.shift(length)

        return mom

    @staticmethod
    def rvi(df, length=10, ema_length=14):
        """Calculate Relative Volatility Index (RVI)."""
        src = df["close"]

        # Calculate standard deviation
        stddev = TechnicalIndicators.stdev(src, length)

        # Calculate price change
        change = src.diff()

        # Upper: EMA of stddev when price goes up (change > 0)
        upper_values = stddev.where(change > 0, 0)
        upper = TechnicalIndicators.ema(upper_values, ema_length)

        # Lower: EMA of stddev when price goes down (change <= 0)
        lower_values = stddev.where(change <= 0, 0)
        lower = TechnicalIndicators.ema(lower_values, ema_length)

        # Calculate RVI
        rvi = 100 * upper / (upper + lower)

        # Handle division by zero
        rvi = rvi.replace([float("inf"), -float("inf")], np.nan)
        rvi = rvi.fillna(50)  # When both upper and lower are 0

        return rvi

    @staticmethod
    def obv(df):
        """Calculate On-Balance Volume (OBV)."""
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

    @staticmethod
    def mfi(df, length=14):
        """Calculate Money Flow Index (MFI)."""
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

    @staticmethod
    def performance_index(df, benchmark_df, period=21):
        """
        Calculate Performance Index against benchmark (S&P 500).
        PI = (Stock Price / Benchmark Price) * (Benchmark MA / Stock MA)
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

    @staticmethod
    def crs(df, benchmark_df, period=21):
        """
        Calculate Comparative Relative Strength (CRS) vs benchmark.
        CRS = Stock Price / Benchmark Price
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
