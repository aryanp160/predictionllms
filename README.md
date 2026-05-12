# Deep Learning for Cryptocurrency Price Prediction
<div float="left">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" height="120" >
  <img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" height="120">
</div>
<div float="right">

</div>

## 

### Introduction
This repo makes use of the state-of-art Deep Learning algorithm to predict the price of Bitcoin, which has the potential to generalize to other cryptocurrency. It leverages models such as CNN and RNN implemented by [Keras](https://github.com/keras-team/keras) running on top of [Tensorflow](https://github.com/tensorflow/tensorflow). You can find more detailed illustration in this [blog post](https://blog.goodaudience.com/predicting-cryptocurrency-price-with-tensorflow-and-keras-e1674b0dc58a).

### Getting Started
This repo has been modernized for Python 3.x and TensorFlow 2.x using Binance historical BTC/USDT data. Install dependencies from `requirements.txt`.

### New End-to-End Pipeline
1. `python data_fetch.py`
   - downloads last 6 months of BTC/USDT OHLCV from Binance for 15m and 1h intervals.
2. `python feature_engineering.py`
   - builds technical indicators and merges 15m/1h features into `data/btc_usdt_15m_features.csv`.
3. `python train_models.py`
   - trains CNN, LSTM, and GRU models, saves checkpoints, scalers, and metrics to `models/`.
4. `python predict_plot.py`
   - loads trained models, produces ensemble forecasts, confidence bands, and saves charts/CSV to `result/`.

### New Features
- **Modern TensorFlow 2.x**: Updated from deprecated TF1 patterns to current TF2/Keras API.
- **Binance API Integration**: Real-time data fetching from Binance instead of deprecated Poloniex.
- **Technical Indicators**: RSI, MACD, EMA, SMA, Bollinger Bands, ATR, VWAP for enhanced feature engineering.
- **Ensemble Predictions**: Combines CNN, LSTM, and GRU forecasts with confidence intervals.
- **Individual Model Forecasts**: Separate scripts for CNN, LSTM, and GRU predictions.
- **Data Timeframe**: Uses last 6 months of 15-minute BTC/USDT data for current market relevance.

### Individual Forecast Scripts
For individual model predictions (faster than full ensemble):
- `python run_gru_forecast.py` → GRU forecast chart and CSV
- `python run_cnn_forecast.py` → CNN forecast chart and CSV  
- `python run_lstm_forecast.py` → LSTM forecast chart and CSV

### Legacy Files
The original `CNN.py`, `LSTM.py`, `GRU.py`, and notebook files remain in the repo for reference, but the new pipeline is the recommended workflow.

### Run
Use the new pipeline for up-to-date predictions:
```
python data_fetch.py
python feature_engineering.py
python train_models.py
python predict_plot.py
```

For individual model forecasts (faster execution):
```
python run_gru_forecast.py
python run_cnn_forecast.py
python run_lstm_forecast.py
```

> The legacy notebooks and old data collection scripts are deprecated. Use Binance API scripts in the new pipeline.
### Input & Output & Loss
The input consists of a list of past Bitcoin data with step size of 256.
The output is the predicted value of the future data with step size of 16. Note that since the data is ticked every five minutes, the input data spans over the past 1280 minutes, while the output cover the future 80 minutes. The datas are scaled with MinMaxScaler provided by sklearn over the entire dataset. The loss is defined as Mean Square Error (MSE).

### Modernized Architecture
The new pipeline uses:
- **Data Source**: Binance REST API for 15m and 1h BTC/USDT candles (last 6 months)
- **Features**: OHLCV + 7 technical indicators (RSI, MACD, EMA, SMA, Bollinger Bands, ATR, VWAP)
- **Models**: CNN (Conv1D), LSTM, GRU with dropout, early stopping, and learning rate scheduling
- **Training**: 30 epochs with validation, best model checkpointing
- **Prediction**: Ensemble averaging with confidence intervals, professional matplotlib charts

### Result
|Model | #Layers  |  Activation    | Validation Loss   |Test Loss (Scale Inverted) |
|----------| ------------- |------|-------| -----|
|   CNN    | 2       | ReLU       |    0.00029     | 114308 |
|   CNN    | 2       | Leaky ReLU       |    0.00029     | 115525 |
|   CNN    | 3       | ReLU       |    0.00029     | 201718 |
|   CNN    | 3       | Leaky ReLU       |    0.00028     | 108700 |
|   CNN    | 4       | ReLU       |    0.00030     | 117947 |
|   CNN    | 4       | Leaky ReLU       |    0.03217     | 12356304 |
|   LSTM    | 1      | tanh + ReLU       |    0.00007     | 26649 |
|   LSTM    | 1      | tanh + Leaky ReLU       |    0.00004     | 15364 |
|   GRU    | 1      | tanh + ReLU       |    0.00004     | 17667 |
|   GRU    | 1      | tanh + Leaky ReLU       |    0.00004     | 15474 |
|   Baseline (Lag)    | -     | -       |    -     | 19122 |
|   Linear Regression   | -     | -       |    -     | 19789 |



Each row of the above table is the model that derives the best validation loss from the total 100 training epochs. From the above result, we can observe that LeakyReLU always seems to yield better loss compared to regular ReLU. However, 4-layered CNN with Leaky ReLU as activation function creates a large validation loss, this can due to wrong deployment of model which might require re-validation. CNN model can be trained very fast (2 seconds/ epoch with GPU), with slightly worse performance than LSTM and GRU. The best model seems to be LSTM with tanh and Leaky ReLU as activation function, though 3-layered CNN seems to be better in capturing local temporal dependency of data.
<div align="center">
	<img src="result/bitcoin2015to2017_close_LSTM_1_tanh_leaky_result.png" width="80%" />
</div>

_LSTM with tanh and Leaky ReLu as activation function._

<div align="center">
	<img src="result/bitcoin2015to2017_close_CNN_3_leaky_result.png" width="80%" />
</div>

_3-layered CNN with Leaky ReLu as activation function._

<div align="center">
	<img src="result/bitcoin2015to2017_close_rw.png" width="80%" />
</div>

_Baseline_

<div align="center">
	<img src="result/bitcoin2015to2017_close_lr.png" width="80%" />
</div>

_Linear Regression_

## Update
Regularization has been done, which can be viewed in PlotRegularization.ipynb.
