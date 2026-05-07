import os

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=1).mean()


def rsi(series: pd.Series, window: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series):
    fast = ema(series, span=12)
    slow = ema(series, span=26)
    macd_line = fast - slow
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line, signal, histogram


def bollinger_bands(series: pd.Series, window: int = 20, multiplier: float = 2.0):
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + multiplier * std
    lower = middle - multiplier * std
    width = upper - lower
    return upper, lower, width


def atr(df: pd.DataFrame, window: int = 14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window, min_periods=1).mean()


def vwap(df: pd.DataFrame):
    price_volume = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
    volume_sum = df['Volume'].rolling(window=14, min_periods=1).sum()
    return price_volume.rolling(window=14, min_periods=1).sum() / (volume_sum + 1e-9)


def build_indicators(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    df = df.copy()
    df[f'{prefix}EMA_12'] = ema(df['Close'], span=12)
    df[f'{prefix}EMA_26'] = ema(df['Close'], span=26)
    df[f'{prefix}SMA_20'] = sma(df['Close'], window=20)
    df[f'{prefix}SMA_50'] = sma(df['Close'], window=50)
    df[f'{prefix}RSI_14'] = rsi(df['Close'], window=14)
    df[f'{prefix}MACD'], df[f'{prefix}MACD_SIGNAL'], df[f'{prefix}MACD_HIST'] = macd(df['Close'])
    df[f'{prefix}BB_UPPER'], df[f'{prefix}BB_LOWER'], df[f'{prefix}BB_WIDTH'] = bollinger_bands(df['Close'], window=20)
    df[f'{prefix}ATR_14'] = atr(df, window=14)
    df[f'{prefix}VWAP'] = vwap(df)
    df[f'{prefix}PRICE_VWAP_DIFF'] = df['Close'] - df[f'{prefix}VWAP']
    df[f'{prefix}BB_PCT'] = (df['Close'] - df[f'{prefix}BB_LOWER']) / (df[f'{prefix}BB_WIDTH'] + 1e-9)
    df[f'{prefix}HL_RANGE'] = df['High'] - df['Low']
    df[f'{prefix}OC_CHANGE'] = df['Close'] - df['Open']
    return df


def merge_timeframes(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    df_15m = df_15m.copy()
    df_1h = df_1h.copy()
    df_1h = df_1h.rename(columns={
        'Open': 'Open_1h',
        'High': 'High_1h',
        'Low': 'Low_1h',
        'Close': 'Close_1h',
        'Volume': 'Volume_1h',
    })
    df_merged = pd.merge_asof(
        df_15m.sort_values('Timestamp'),
        df_1h.sort_values('Timestamp'),
        left_on='Timestamp',
        right_on='Timestamp',
        direction='backward',
    )
    return df_merged


def prepare_feature_dataset():
    os.makedirs('data', exist_ok=True)
    df_15m = pd.read_csv('data/btc_usdt_15m.csv', parse_dates=['Timestamp'])
    df_1h = pd.read_csv('data/btc_usdt_1h.csv', parse_dates=['Timestamp'])

    latest_start = max(df_15m['Timestamp'].min(), df_1h['Timestamp'].min())
    cutoff = df_15m['Timestamp'].max() - pd.Timedelta(days=183)
    df_15m = df_15m[df_15m['Timestamp'] >= cutoff].copy()
    df_1h = df_1h[df_1h['Timestamp'] >= cutoff].copy()

    df_15m = build_indicators(df_15m, prefix='M15_')
    df_1h = build_indicators(df_1h, prefix='H1_')
    df_merged = merge_timeframes(df_15m, df_1h)
    df_merged = df_merged.dropna().reset_index(drop=True)
    output_path = os.path.join('data', 'btc_usdt_15m_features.csv')
    df_merged.to_csv(output_path, index=False)
    print(f'Feature dataset saved to {output_path} ({len(df_merged)} rows)')


if __name__ == '__main__':
    prepare_feature_dataset()
