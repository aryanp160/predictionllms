import csv
import json
import os
import time
import urllib.request
import datetime

import pandas as pd


BINANCE_KLINES_URL = 'https://api.binance.com/api/v3/klines'


def fetch_klines(symbol: str, interval: str, start_time_ms: int, end_time_ms: int, limit: int = 1000):
    candles = []
    next_start = start_time_ms

    while next_start < end_time_ms:
        params = [
            f'symbol={symbol}',
            f'interval={interval}',
            f'startTime={next_start}',
            f'endTime={end_time_ms}',
            f'limit={limit}',
        ]
        url = BINANCE_KLINES_URL + '?' + '&'.join(params)
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())

        if not data:
            break

        candles.extend(data)
        next_start = data[-1][0] + 1
        if len(data) < limit:
            break
        time.sleep(0.15)

    return candles


def build_dataframe(candles):
    df = pd.DataFrame(
        candles,
        columns=[
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteAssetVolume', 'Trades',
            'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
        ]
    )
    df = df[['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime']]
    df['Timestamp'] = pd.to_datetime(df['CloseTime'], unit='ms')
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def download_symbol_history(symbol: str, intervals: list[str], months: int = 6):
    os.makedirs('data', exist_ok=True)
    end_time_ms = int(time.time() * 1000)
    start_time_ms = int((datetime.datetime.utcnow() - datetime.timedelta(days=months * 30)).timestamp() * 1000)

    for interval in intervals:
        print(f'Fetching {symbol} {interval} data for last {months} months...')
        candles = fetch_klines(symbol, interval, start_time_ms, end_time_ms)
        df = build_dataframe(candles)
        filename = os.path.join('data', f'btc_usdt_{interval}.csv')
        df.to_csv(filename, index=False)
        print(f'  saved {len(df)} rows to {filename}')


if __name__ == '__main__':
    download_symbol_history('BTCUSDT', ['15m', '1h'], months=6)
