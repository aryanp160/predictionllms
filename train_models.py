import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Dense, Dropout, GlobalAveragePooling1D,
                                     GRU, LSTM)
from tensorflow.keras.models import Sequential


DATA_PATH = 'data/btc_usdt_15m_features.csv'


def load_feature_data(path: str = DATA_PATH):
    df = pd.read_csv(path, parse_dates=['Timestamp'])
    features = df.drop(columns=['Timestamp']).copy()
    column_names = features.columns.tolist()
    return features, column_names, df['Timestamp']


def create_sequences(data: np.ndarray, target: np.ndarray, seq_len: int = 64, horizon: int = 16):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len:i + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def evaluation_metrics(true: np.ndarray, pred: np.ndarray):
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)
    mae = mean_absolute_error(true_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
    mape = np.mean(np.abs((true_flat - pred_flat) / (true_flat + 1e-9))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def build_cnn_model(input_shape, output_size):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_size, activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_lstm_model(input_shape, output_size):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    attention = Dense(1, activation='tanh')(x)
    attention = Activation('softmax', name='attention_weights')(attention)
    x = x * attention
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(output_size, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def build_gru_model(input_shape, output_size):
    model = Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.25),
        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.25),
        GRU(32, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_size, activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def plot_training_curves(history, name: str):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(history.history['loss'], label='train loss')
    ax.plot(history.history['val_loss'], label='val loss')
    ax.set_title(f'{name} Training Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    plt.tight_layout()
    os.makedirs('result', exist_ok=True)
    fig.savefig(f'result/{name}_loss.png')
    plt.close(fig)


def save_metrics(metrics: dict, name: str):
    os.makedirs('result', exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(f'result/{name}_metrics.csv', index=False)


def train_and_persist(model, name: str, X_train, y_train, X_val, y_val):
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        f'models/{name}.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[checkpoint, early_stopping, lr_scheduler],
        verbose=2,
    )
    plot_training_curves(history, name)
    return model, history


def main():
    features, feature_columns, timestamps = load_feature_data()
    target = features['Close'].values
    X_all, y_all = create_sequences(features.values, target, seq_len=64, horizon=16)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_all, y_all, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.17647, shuffle=False)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_train_shape = X_train.shape
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = feature_scaler.fit_transform(X_train_flat).reshape(X_train_shape)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open('models/target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)

    input_shape = X_train_scaled.shape[1:]
    output_size = y_train_scaled.shape[1]

    cnn = build_cnn_model(input_shape, output_size)
    cnn, _ = train_and_persist(cnn, 'cnn_model', X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    cnn_metrics = evaluation_metrics(y_test, target_scaler.inverse_transform(cnn.predict(X_test_scaled)))
    save_metrics(cnn_metrics, 'cnn')

    lstm = build_lstm_model(input_shape, output_size)
    lstm, _ = train_and_persist(lstm, 'lstm_model', X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    lstm_metrics = evaluation_metrics(y_test, target_scaler.inverse_transform(lstm.predict(X_test_scaled)))
    save_metrics(lstm_metrics, 'lstm')

    gru = build_gru_model(input_shape, output_size)
    gru, _ = train_and_persist(gru, 'gru_model', X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    gru_metrics = evaluation_metrics(y_test, target_scaler.inverse_transform(gru.predict(X_test_scaled)))
    save_metrics(gru_metrics, 'gru')

    history = {
        'cnn_metrics': cnn_metrics,
        'lstm_metrics': lstm_metrics,
        'gru_metrics': gru_metrics,
        'feature_columns': feature_columns,
    }
    pd.DataFrame([history]).to_json('result/training_summary.json', orient='records', indent=2)
    print('Training complete and saved results to result/ and models/')


if __name__ == '__main__':
    main()
