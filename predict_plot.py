import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


DATA_PATH = 'data/btc_usdt_15m_features.csv'
MODEL_PATHS = {
    'cnn': 'models/cnn_model.keras',
    'lstm': 'models/lstm_model.keras',
    'gru': 'models/gru_model.keras',
}


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
    df = df.dropna().reset_index(drop=True)
    return df


def build_sequence(df: pd.DataFrame, seq_len: int = 64):
    feature_cols = [c for c in df.columns if c != 'Timestamp']
    data = df[feature_cols].values.astype(np.float32)
    return data[-seq_len:][None, ...], feature_cols


def load_scalers():
    with open('models/feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    with open('models/target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    return feature_scaler, target_scaler


def ensemble_confidence(predictions: dict):
    stacked = np.stack(list(predictions.values()), axis=0)
    mean_preds = np.mean(stacked, axis=0)
    std_preds = np.std(stacked, axis=0)
    interval = 1.96 * std_preds
    confidence = 1.0 - np.clip(std_preds / (np.abs(mean_preds) + 1e-9), 0, 1)
    confidence_score = float(np.clip(confidence.mean(), 0, 1))
    return mean_preds, interval, confidence_score


def forecast():
    df = load_data()
    last_close = df['Close'].iloc[-1]
    last_timestamp = df['Timestamp'].iloc[-1]
    prediction_window = 16
    horizon_1h = 4

    seq, feature_cols = build_sequence(df)
    feature_scaler, target_scaler = load_scalers()
    seq_scaled = feature_scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape)

    predictions = {}
    for name, path in MODEL_PATHS.items():
        model = load_model(path)
        pred_scaled = model.predict(seq_scaled, verbose=0)
        predictions[name] = target_scaler.inverse_transform(pred_scaled).reshape(-1)

    ensemble_mean, ensemble_interval, confidence_score = ensemble_confidence(predictions)
    future_times = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=15), periods=prediction_window, freq='15min')

    pct_1h = (ensemble_mean[horizon_1h - 1] - last_close) / (last_close + 1e-9) * 100
    pct_4h = (ensemble_mean[prediction_window - 1] - last_close) / (last_close + 1e-9) * 100
    direction_1h = 'Bullish' if pct_1h >= 0 else 'Bearish'
    direction_4h = 'Bullish' if pct_4h >= 0 else 'Bearish'

    output = pd.DataFrame({
        'Timestamp': future_times,
        'Ensemble': ensemble_mean,
        'LowerBound': ensemble_mean - ensemble_interval,
        'UpperBound': ensemble_mean + ensemble_interval,
    })
    output.to_csv('result/forecast_output.csv', index=False)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    ax.plot(df['Timestamp'].iloc[-200:], df['Close'].iloc[-200:], color='#1f77b4', linewidth=1.6, label='Actual Close')
    ax.plot(df['Timestamp'].iloc[-200:], df['Close'].rolling(8).mean().iloc[-200:], color='#1f77b4', alpha=0.35, linewidth=2, label='Actual Smooth')

    for name, preds in predictions.items():
        ax.plot(future_times, preds, linestyle='--', linewidth=2, label=f'{name.upper()} Forecast')

    ax.plot(future_times, ensemble_mean, linestyle='-', linewidth=3, color='#d62728', label='Ensemble Forecast')
    ax.fill_between(future_times, ensemble_mean - ensemble_interval, ensemble_mean + ensemble_interval, color='#d62728', alpha=0.15, label='Confidence Interval')
    ax.axvspan(future_times[0], future_times[-1], color='#f2f2f2', alpha=0.5)

    ax.set_title('BTC/USDT Forecast: Actual Price with CNN, LSTM, GRU and Ensemble Predictions', fontsize=16)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(df['Timestamp'].iloc[-200], future_times[-1])

    annotation = (
        f'Last Close: {last_close:.2f} USDT\n'
        f'1h Forecast: {ensemble_mean[horizon_1h - 1]:.2f} USDT ({pct_1h:+.2f}% / {direction_1h})\n'
        f'4h Forecast: {ensemble_mean[-1]:.2f} USDT ({pct_4h:+.2f}% / {direction_4h})\n'
        f'\nEnsemble Confidence: {confidence_score * 100:.1f}%'
    )
    ax.text(0.01, 0.97, annotation, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    os.makedirs('result', exist_ok=True)
    fig.savefig('result/forecast_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved forecast chart to result/forecast_comparison.png')
    print('Saved forecast CSV to result/forecast_output.csv')


if __name__ == '__main__':
    forecast()
