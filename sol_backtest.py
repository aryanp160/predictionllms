# ============================================================
# SOL/USDT BINANCE-STYLE FUTURES BACKTEST
# ============================================================
#
# FEATURES
# ------------------------------------------------------------
# - Binance FREE API
# - RSI + STOCH strategy
# - EMA trend filtering
# - Binance-style TP/SL behavior
# - 15x leverage
# - 2% fees
# - Equity curve
# - Trade statistics
#
# ============================================================

# INSTALL:
# pip install pandas numpy matplotlib requests ta

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

LOOKBACK_DAYS = 60

CAPITAL = 1000

LEVERAGE = 15


TP_PERCENT = 7
SL_PERCENT = 7

FEES_PERCENT = 2

# ============================================================
# DOWNLOAD DATA
# ============================================================

def get_binance_data():

    print("Downloading Binance data...")

    url = "https://api.binance.com/api/v3/klines"

    all_data = []

    end_time = int(datetime.now().timestamp() * 1000)

    start_time = int(
        (
            datetime.now()
            -
            timedelta(days=LOOKBACK_DAYS)
        ).timestamp() * 1000
    )

    while start_time < end_time:

        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": start_time,
            "limit": 1000
        }

        response = requests.get(url, params=params)

        data = response.json()

        if len(data) == 0:
            break

        all_data.extend(data)

        # move to next candle
        start_time = data[-1][0] + 1

        print(f"Downloaded {len(all_data)} candles...")

    df = pd.DataFrame(all_data, columns=[
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

    df['time'] = pd.to_datetime(
        df['time'],
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
        df[col] = pd.to_numeric(df[col])

    print(f"\nFINAL CANDLES: {len(df)}")

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

    # TREND
    trend = (
        row['EMA50']
        >
        row['EMA200']
    )

    # RSI RECOVERY
    rsi = (
        prev['RSI'] < 35
        and
        row['RSI'] > prev['RSI']
    )

    # STOCH CROSS UP
    stoch = (
        prev['K'] < prev['D']
        and
        row['K'] > row['D']
        and
        row['K'] < 25
    )

    # BULLISH CANDLE
    candle = (
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
        candle
    )

# ============================================================
# SHORT CONDITION
# ============================================================

def short_condition(df, i):

    row = df.iloc[i]

    prev = df.iloc[i - 1]

    # DOWNTREND
    trend = (
        row['EMA50']
        <
        row['EMA200']
    )

    # RSI WEAKENING
    rsi = (
        prev['RSI'] > 65
        and
        row['RSI'] < prev['RSI']
    )

    # STOCH CROSS DOWN
    stoch = (
        prev['K'] > prev['D']
        and
        row['K'] < row['D']
        and
        row['K'] > 75
    )

    # BEARISH CANDLE
    candle = (
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
        candle
    )

# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest(df):

    print("Running backtest...")

    balance = 0

    equity_curve = []

    trades = []

    in_trade = False

    trade_type = None

    entry_price = 0

    tp_price = 0

    sl_price = 0

    for i in range(200, len(df)):

        row = df.iloc[i]

        # ====================================================
        # ENTRY
        # ====================================================

        if not in_trade:

            # LONG ENTRY
            if long_condition(df, i):

                in_trade = True

                trade_type = "LONG"

                entry_price = row['close']

                # BINANCE STYLE
                tp_move = TP_PERCENT / LEVERAGE
                sl_move = SL_PERCENT / LEVERAGE

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

                print(
                    f"LONG ENTRY: "
                    f"{round(entry_price,2)}"
                )

            # SHORT ENTRY
            elif short_condition(df, i):

                in_trade = True

                trade_type = "SHORT"

                entry_price = row['close']

                # BINANCE STYLE
                tp_move = TP_PERCENT / LEVERAGE
                sl_move = SL_PERCENT / LEVERAGE

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

                print(
                    f"SHORT ENTRY: "
                    f"{round(entry_price,2)}"
                )

        # ====================================================
        # MANAGE TRADES
        # ====================================================

        else:

            current_price = row['close']

            # =================================================
            # LONG
            # =================================================

            if trade_type == "LONG":

                # TAKE PROFIT
                if current_price >= tp_price:

                    gross_profit = (
                        CAPITAL
                        *
                        (TP_PERCENT / 100)
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

                    trades[-1]['exit'] = current_price

                    trades[-1]['pnl'] = net_profit

                    print(
                        f"LONG TP | "
                        f"+₹{round(net_profit,2)}"
                    )

                    in_trade = False

                # STOP LOSS
                elif current_price <= sl_price:

                    gross_loss = (
                        CAPITAL
                        *
                        (SL_PERCENT / 100)
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

                    trades[-1]['exit'] = current_price

                    trades[-1]['pnl'] = -total_loss

                    print(
                        f"LONG SL | "
                        f"-₹{round(total_loss,2)}"
                    )

                    in_trade = False

            # =================================================
            # SHORT
            # =================================================

            elif trade_type == "SHORT":

                # TAKE PROFIT
                if current_price <= tp_price:

                    gross_profit = (
                        CAPITAL
                        *
                        (TP_PERCENT / 100)
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

                    trades[-1]['exit'] = current_price

                    trades[-1]['pnl'] = net_profit

                    print(
                        f"SHORT TP | "
                        f"+₹{round(net_profit,2)}"
                    )

                    in_trade = False

                # STOP LOSS
                elif current_price >= sl_price:

                    gross_loss = (
                        CAPITAL
                        *
                        (SL_PERCENT / 100)
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

                    trades[-1]['exit'] = current_price

                    trades[-1]['pnl'] = -total_loss

                    print(
                        f"SHORT SL | "
                        f"-₹{round(total_loss,2)}"
                    )

                    in_trade = False

        equity_curve.append(balance)

    return trades, equity_curve

# ============================================================
# PLOT RESULTS
# ============================================================

def plot_results(equity_curve):

    plt.figure(figsize=(15, 7))

    plt.plot(
        equity_curve,
        linewidth=2
    )

    plt.title(
        "SOL/USDT Binance-Style Futures Strategy"
    )

    plt.xlabel("Trades")

    plt.ylabel("Profit (₹)")

    plt.grid(True)

    plt.show()

# ============================================================
# MAIN
# ============================================================

print("====================================")
print("SOL/USDT BINANCE FUTURES BACKTEST")
print("====================================")

# DOWNLOAD DATA
df = get_binance_data()

# ADD INDICATORS
df = add_indicators(df)

# RUN BACKTEST
trades, equity_curve = backtest(df)

# RESULTS
trades_df = pd.DataFrame(trades)
# ==========================================
# LONG / SHORT ANALYSIS
# ==========================================

long_trades = trades_df[
    trades_df['type'] == 'LONG'
]

short_trades = trades_df[
    trades_df['type'] == 'SHORT'
]

# LONG STATS
long_profit = long_trades['pnl'].sum()

long_wins = len(
    long_trades[
        long_trades['pnl'] > 0
    ]
)

long_losses = len(
    long_trades[
        long_trades['pnl'] < 0
    ]
)

# SHORT STATS
short_profit = short_trades['pnl'].sum()

short_wins = len(
    short_trades[
        short_trades['pnl'] > 0
    ]
)

short_losses = len(
    short_trades[
        short_trades['pnl'] < 0
    ]
)

print("\n========== LONG STATS ==========")

print(
    f"Total Long Trades: "
    f"{len(long_trades)}"
)

print(
    f"Long Wins: "
    f"{long_wins}"
)

print(
    f"Long Losses: "
    f"{long_losses}"
)

print(
    f"Long Profit: "
    f"₹{round(long_profit,2)}"
)

print("\n========== SHORT STATS ==========")

print(
    f"Total Short Trades: "
    f"{len(short_trades)}"
)

print(
    f"Short Wins: "
    f"{short_wins}"
)

print(
    f"Short Losses: "
    f"{short_losses}"
)

print(
    f"Short Profit: "
    f"₹{round(short_profit,2)}"
)
print("\n====================================")
print("FINAL RESULTS")
print("====================================")

print(f"Total Trades: {len(trades_df)}")

if len(trades_df) > 0:

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
    ) * 100

    total_profit = trades_df['pnl'].sum()

    avg_trade = trades_df['pnl'].mean()

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

    print(
        f"Average Trade: "
        f"₹{round(avg_trade,2)}"
    )

# PLOT RESULTS
plot_results(equity_curve)