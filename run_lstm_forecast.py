import os
import time
import json
import urllib.request
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Input


def fetch_binance_klines(symbol, interval, start_time_ms, end_time_ms=None, limit=1000):
    rows = []
    base_url = 'https://api.binance.com/api/v3/klines'
    while True:
        params = [
            f'symbol={symbol}',
            f'interval={interval}',
            f'startTime={start_time_ms}',
            f'limit={limit}',
        ]
        if end_time_ms is not None:
            params.append(f'endTime={end_time_ms}')
        url = base_url + '?' + '&'.join(params)

        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())

        if not data:
            break

        rows.extend(data)
        last_time = data[-1][0]

        if len(data) < limit:
            break

        start_time_ms = last_time + 1
        time.sleep(0.2)

    return rows


class PastSampler:
    def __init__(self, N, K, sliding_window=True):
        self.N = N
        self.K = K
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0] % M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1, 1)
            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(-1, 1)

        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]
        return B[:, :ci], B[:, ci:]


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    symbol = 'BTCUSDT'
    interval = '15m'
    start_time = int(datetime.datetime(2022, 1, 1, 0, 0).timestamp() * 1000)
    end_time = int(time.time() * 1000)
    csv_path = os.path.join('data', 'btc_usdt_15m_2022_present.csv')

    if os.path.exists(csv_path):
        print('Loading cached Binance CSV data from', csv_path)
        df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    else:
        print('Fetching Binance data from 2022 to present...')
        klines = fetch_binance_klines(symbol, interval, start_time, end_time)
        print(f'Fetched {len(klines)} candles')

        df = pd.DataFrame(
            klines,
            columns=[
                'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                'CloseTime', 'QuoteAssetVolume', 'Trades',
                'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
            ]
        )
        df['Timestamp'] = pd.to_datetime(df['CloseTime'], unit='ms')
        df = df[['Open', 'High', 'Low', 'Close', 'Timestamp']]
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
        df.to_csv(csv_path, index=False)
        print('Saved Binance data to', csv_path)

    window_N = 128
    forecast_K = 16

    price_data = df[['Close']].values.astype(np.float32)
    price_data = price_data.reshape(-1, 1, 1)

    sampler = PastSampler(N=window_N, K=forecast_K, sliding_window=True)
    X, y = sampler.transform(price_data)

    if X.shape[0] > 30000:
        X = X[-30000:]
        y = y[-30000:]

    scaler = MinMaxScaler()
    X_flat = X.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    scaler.fit(np.concatenate([X_flat, y_flat], axis=0))

    X_scaled = scaler.transform(X_flat).reshape(X.shape)
    y_scaled = scaler.transform(y_flat).reshape(y.shape)

    split = int(0.8 * X_scaled.shape[0])
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y_scaled[:split], y_scaled[split:]

    model = Sequential([
        Input(shape=(window_N, 1)),
        LSTM(64, return_sequences=False),
        Activation('tanh'),
        Dropout(0.2),
        Dense(forecast_K),
        Activation('linear')
    ])
    model.compile(loss='mse', optimizer='adam')

    print('Training LSTM model...')
    model.fit(
        X_train,
        y_train.reshape(y_train.shape[0], forecast_K),
        validation_data=(X_val, y_val.reshape(y_val.shape[0], forecast_K)),
        epochs=6,
        batch_size=64,
        verbose=2,
    )

    last_input = X_scaled[-1:]
    pred_scaled = model.predict(last_input)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

    recent_df = df.iloc[-300:].copy()
    last_time = recent_df['Timestamp'].iloc[-1]
    future_times = pd.date_range(last_time + pd.Timedelta(minutes=15), periods=forecast_K, freq='15min')

    plt.figure(figsize=(14, 7))
    plt.plot(recent_df['Timestamp'], recent_df['Close'], label='Actual Close', color='blue')
    plt.plot(future_times, pred, label='LSTM Forecast', color='orange', linestyle='--', linewidth=2)
    plt.axvline(last_time, color='gray', linestyle=':', linewidth=1)
    plt.scatter(future_times, pred, color='orange')
    plt.title('BTC/USDT 15m Close Price and LSTM Forecast')
    plt.xlabel('Time')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    chart_path = os.path.join('result', 'lstm_forecast.png')
    plt.savefig(chart_path)
    print('Saved forecast chart to', chart_path)

    pred_df = pd.DataFrame({'Timestamp': future_times, 'PredictedClose': pred})
    pred_csv = os.path.join('result', 'lstm_forecast_prediction.csv')
    pred_df.to_csv(pred_csv, index=False)
    print('Saved forecast data to', pred_csv)
