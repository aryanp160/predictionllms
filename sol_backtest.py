

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator

from datetime import datetime, timedelta

# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "ETHUSDT"

INTERVAL = "5m"

LOOKBACK_DAYS = 365

STARTING_CAPITAL = 1500

LEVERAGE = 25

# ============================================================
# ENABLE / DISABLE
# ============================================================

ENABLE_LONGS = False

ENABLE_SHORTS = True

# ============================================================
# LONG SETTINGS
# ============================================================

LONG_TP_PERCENT = 10
LONG_SL_PERCENT = 7

# ============================================================
# SHORT SETTINGS
# ============================================================

SHORT_TP_PERCENT = 20
SHORT_SL_PERCENT = 5

# ============================================================
# FEES
# ============================================================

FEES_PERCENT = 2

# ============================================================
# DATABASE FILE
# ============================================================

DATA_FILE = f"{SYMBOL}_{INTERVAL}_candles.csv"

# ============================================================
# DOWNLOAD / LOAD DATA
# ============================================================

def get_binance_data():

    print("====================================")
    print("LOADING MARKET DATA")
    print("====================================")

    url = "https://api.binance.com/api/v3/klines"

    # LOAD EXISTING DATA
    if os.path.exists(DATA_FILE):

        print("Loading saved candles...")

        existing_df = pd.read_csv(DATA_FILE)

        existing_df['time'] = pd.to_datetime(
            existing_df['time']
        )

        print(
            f"Saved candles: "
            f"{len(existing_df)}"
        )

        last_timestamp = int(
            pd.Timestamp(
                existing_df['time'].iloc[-1]
            ).timestamp() * 1000
        )

        start_time = last_timestamp + 1

    else:

        print("No saved candle database found.")

        existing_df = pd.DataFrame()

        start_time = int(
            (
                datetime.now()
                -
                timedelta(days=LOOKBACK_DAYS)
            ).timestamp() * 1000
        )

    end_time = int(
        datetime.now().timestamp() * 1000
    )

    all_new_data = []

    # DOWNLOAD ONLY NEW DATA
    while start_time < end_time:

        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": start_time,
            "limit": 1000
        }

        response = requests.get(
            url,
            params=params
        )

        data = response.json()

        if len(data) == 0:
            break

        all_new_data.extend(data)

        start_time = data[-1][0] + 1

        print(
            f"Downloaded "
            f"{len(all_new_data)} new candles..."
        )

    # NO NEW DATA
    if len(all_new_data) == 0:

        print("\nNo new candles needed.")

        return existing_df

    # CREATE DATAFRAME
    new_df = pd.DataFrame(all_new_data, columns=[
        'time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_volume',
        'trades',
        'tb_base',
        'tb_quote',
        'ignore'
    ])

    new_df['time'] = pd.to_datetime(
        new_df['time'],
        unit='ms'
    )

    numeric_columns = [
        'open',
        'high',
        'low',
        'close',
        'volume'
    ]

    for col in numeric_columns:

        new_df[col] = pd.to_numeric(
            new_df[col]
        )

    # MERGE OLD + NEW
    if len(existing_df) > 0:

        df = pd.concat([
            existing_df,
            new_df
        ])

        df = df.drop_duplicates(
            subset=['time']
        )

    else:

        df = new_df

    # SORT
    df = df.sort_values(
        by='time'
    )

    # SAVE
    df.to_csv(
        DATA_FILE,
        index=False
    )

    print("\n====================================")
    print("DATABASE UPDATED")
    print("====================================")

    print(
        f"Total candles saved: "
        f"{len(df)}"
    )

    return df

# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    print("Adding indicators...")

    # RSI
    rsi = RSIIndicator(
        close=df['close'],
        window=14
    )

    df['RSI'] = rsi.rsi()

    # STOCHASTIC
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        smooth_window=3
    )

    df['K'] = stoch.stoch()

    df['D'] = stoch.stoch_signal()

    # EMA50
    df['EMA50'] = (
        df['close']
        .ewm(span=50)
        .mean()
    )

    # EMA200
    df['EMA200'] = (
        df['close']
        .ewm(span=200)
        .mean()
    )

    return df

# ============================================================
# LONG CONDITION
# ============================================================

def long_condition(df, i):

    row = df.iloc[i]

    prev = df.iloc[i - 1]

    trend = (
        row['EMA50']
        >
        row['EMA200']
    )

    rsi = (
        prev['RSI'] < 35
        and
        row['RSI'] > prev['RSI']
    )

    stoch = (
        prev['K'] < prev['D']
        and
        row['K'] > row['D']
        and
        row['K'] < 25
    )

    bullish = (
        row['close']
        >
        row['open']
    )

    return (
        trend
        and
        rsi
        and
        stoch
        and
        bullish
    )

# ============================================================
# SHORT CONDITION
# ============================================================

def short_condition(df, i):

    row = df.iloc[i]

    prev = df.iloc[i - 1]

    trend = (
        row['EMA50']
        <
        row['EMA200']
    )

    rsi = (
        prev['RSI'] > 65
        and
        row['RSI'] < prev['RSI']
    )

    stoch = (
        prev['K'] > prev['D']
        and
        row['K'] < row['D']
        and
        row['K'] > 75
    )

    bearish = (
        row['close']
        <
        row['open']
    )

    return (
        trend
        and
        rsi
        and
        stoch
        and
        bearish
    )

# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest(df):

    print("Running backtest...")

    balance = STARTING_CAPITAL

    equity_curve = []

    equity_dates = []

    trades = []

    in_trade = False

    trade_type = None

    for i in range(200, len(df)):

        row = df.iloc[i]

        # ====================================================
        # 50% COMPOUNDING
        # ====================================================

        trade_capital = balance * 0.8

        # ====================================================
        # ENTRY
        # ====================================================

        if not in_trade:

            # LONG
            if ENABLE_LONGS and long_condition(df, i):

                in_trade = True

                trade_type = "LONG"

                entry_price = row['close']

                tp_move = (
                    LONG_TP_PERCENT
                    / LEVERAGE
                )

                sl_move = (
                    LONG_SL_PERCENT
                    / LEVERAGE
                )

                tp_price = (
                    entry_price
                    *
                    (1 + tp_move / 100)
                )

                sl_price = (
                    entry_price
                    *
                    (1 - sl_move / 100)
                )

                trades.append({
                    "time": row['time'],
                    "type": "LONG",
                    "entry": entry_price
                })

            # SHORT
            elif ENABLE_SHORTS and short_condition(df, i):

                in_trade = True

                trade_type = "SHORT"

                entry_price = row['close']

                tp_move = (
                    SHORT_TP_PERCENT
                    / LEVERAGE
                )

                sl_move = (
                    SHORT_SL_PERCENT
                    / LEVERAGE
                )

                tp_price = (
                    entry_price
                    *
                    (1 - tp_move / 100)
                )

                sl_price = (
                    entry_price
                    *
                    (1 + sl_move / 100)
                )

                trades.append({
                    "time": row['time'],
                    "type": "SHORT",
                    "entry": entry_price
                })

        # ====================================================
        # TRADE MANAGEMENT
        # ====================================================

        else:

            current_price = row['close']

            # LONG
            if trade_type == "LONG":

                # TP
                if current_price >= tp_price:

                    gross_profit = (
                        trade_capital
                        *
                        (LONG_TP_PERCENT / 100)
                    )

                    fees = (
                        gross_profit
                        *
                        (FEES_PERCENT / 100)
                    )

                    net_profit = (
                        gross_profit
                        -
                        fees
                    )

                    balance += net_profit

                    trades[-1]['pnl'] = net_profit

                    in_trade = False

                # SL
                elif current_price <= sl_price:

                    gross_loss = (
                        trade_capital
                        *
                        (LONG_SL_PERCENT / 100)
                    )

                    fees = (
                        gross_loss
                        *
                        (FEES_PERCENT / 100)
                    )

                    total_loss = (
                        gross_loss
                        +
                        fees
                    )

                    balance -= total_loss

                    trades[-1]['pnl'] = -total_loss

                    in_trade = False

            # SHORT
            elif trade_type == "SHORT":

                # TP
                if current_price <= tp_price:

                    gross_profit = (
                        trade_capital
                        *
                        (SHORT_TP_PERCENT / 100)
                    )

                    fees = (
                        gross_profit
                        *
                        (FEES_PERCENT / 100)
                    )

                    net_profit = (
                        gross_profit
                        -
                        fees
                    )

                    balance += net_profit

                    trades[-1]['pnl'] = net_profit

                    in_trade = False

                # SL
                elif current_price >= sl_price:

                    gross_loss = (
                        trade_capital
                        *
                        (SHORT_SL_PERCENT / 100)
                    )

                    fees = (
                        gross_loss
                        *
                        (FEES_PERCENT / 100)
                    )

                    total_loss = (
                        gross_loss
                        +
                        fees
                    )

                    balance -= total_loss

                    trades[-1]['pnl'] = -total_loss

                    in_trade = False

        equity_curve.append(balance)

        equity_dates.append(row['time'])

    return trades, equity_curve, equity_dates, balance

# ============================================================
# PLOT RESULTS
# ============================================================

def plot_results(df, equity_curve, equity_dates):

    # EQUITY CURVE
    plt.figure(figsize=(15, 7))

    plt.plot(
        equity_dates,
        equity_curve,
        linewidth=2
    )

    plt.title(
        "Strategy Equity Curve"
    )

    plt.xlabel("Date")

    plt.ylabel("Balance (₹)")

    plt.grid(True)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    # ETH PRICE
    plt.figure(figsize=(15, 7))

    plt.plot(
        df['time'],
        df['close'],
        linewidth=1
    )

    plt.title(
        "ETH/USDT Price"
    )

    plt.xlabel("Date")

    plt.ylabel("ETH Price")

    plt.grid(True)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

# ============================================================
# MAIN
# ============================================================

print("====================================")
print("ETH/USDT FUTURES BACKTEST")
print("====================================")

df = get_binance_data()

df = add_indicators(df)

trades, equity_curve, equity_dates, balance = backtest(df)

trades_df = pd.DataFrame(trades)

# ============================================================
# RESULTS
# ============================================================

long_trades = trades_df[
    trades_df['type'] == 'LONG'
]

short_trades = trades_df[
    trades_df['type'] == 'SHORT'
]

print("\n========== LONG STATS ==========")

print(f"Total Long Trades: {len(long_trades)}")

print(
    f"Long Profit: "
    f"₹{round(long_trades['pnl'].sum(),2)}"
)

print("\n========== SHORT STATS ==========")

print(f"Total Short Trades: {len(short_trades)}")

print(
    f"Short Profit: "
    f"₹{round(short_trades['pnl'].sum(),2)}"
)

print("\n====================================")
print("FINAL RESULTS")
print("====================================")

wins = len(
    trades_df[
        trades_df['pnl'] > 0
    ]
)

losses = len(
    trades_df[
        trades_df['pnl'] < 0
    ]
)

winrate = (
    wins / len(trades_df)
) * 100 if len(trades_df) > 0 else 0

total_profit = (
    trades_df['pnl'].sum()
)

print(f"Starting Capital: ₹{STARTING_CAPITAL}")

print(
    f"Final Balance: "
    f"₹{round(balance,2)}"
)

print(
    f"Total Trades: "
    f"{len(trades_df)}"
)

print(f"Wins: {wins}")

print(f"Losses: {losses}")

print(
    f"Win Rate: "
    f"{round(winrate,2)}%"
)

print(
    f"Total Profit: "
    f"₹{round(total_profit,2)}"
)

# ============================================================
# PLOT
# ============================================================

plot_results(
    df,
    equity_curve,
    equity_dates
)