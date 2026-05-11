# ============================================================
# ETH/USDT ADVANCED SMART FUTURES BACKTEST
# ============================================================
#
# FEATURES
# ------------------------------------------------------------
# ✅ Binance API
# ✅ Full historical candles
# ✅ RSI + STOCH strategy
# ✅ EMA trend filtering
# ✅ ATR volatility filter
# ✅ Volume confirmation
# ✅ Dynamic compounding
# ✅ Binance-style TP/SL
# ✅ Long / Short analysis
# ✅ Equity curve
#
# ============================================================

# INSTALL:
# pip install pandas matplotlib requests ta

import requests
import pandas as pd
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator

from ta.volatility import AverageTrueRange

from datetime import datetime, timedelta

# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "ETHUSDT"

INTERVAL = "5m"

LOOKBACK_DAYS = 365

STARTING_CAPITAL = 1000

LEVERAGE = 15

# BINANCE STYLE
# Account percentage returns

TP_PERCENT = 7
SL_PERCENT = 14

# REALISTIC BINANCE FEES
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

        response = requests.get(
            url,
            params=params
        )

        data = response.json()

        if len(data) == 0:
            break

        all_data.extend(data)

        start_time = data[-1][0] + 1

        print(
            f"Downloaded "
            f"{len(all_data)} candles..."
        )

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

    numeric_cols = [
        'open',
        'high',
        'low',
        'close',
        'volume'
    ]

    for col in numeric_cols:
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

    # VOLUME SMA
    df['VOL_SMA'] = (
        df['volume']
        .rolling(20)
        .mean()
    )

    # ATR
    atr = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )

    df['ATR'] = (
        atr.average_true_range()
    )

    # ATR MEAN
    df['ATR_MEAN'] = (
        df['ATR']
        .rolling(50)
        .mean()
    )

    return df

# ============================================================
# LONG CONDITION
# ============================================================

def long_condition(df, i):

    row = df.iloc[i]

    prev = df.iloc[i - 1]

    # STRONG TREND
    trend = (
        row['EMA50']
        >
        row['EMA200'] * 1.002
    )

    # RSI RECOVERY
    rsi = (
        prev['RSI'] < 35
        and
        row['RSI'] > prev['RSI']
    )

    # STOCH CROSS
    stoch = (
        prev['K'] < prev['D']
        and
        row['K'] > row['D']
        and
        row['K'] < 25
    )

    # BULLISH CANDLE
    bullish = (
        row['close']
        >
        row['open']
    )

    # VOLUME CONFIRMATION
    volume = (
        row['volume']
        >
        row['VOL_SMA']
    )

    # VOLATILITY FILTER
    volatility = (
        row['ATR']
        >
        row['ATR_MEAN']
    )

    return (
        trend
        and
        rsi
        and
        stoch
        and
        bullish
        and
        volume
        and
        volatility
    )

# ============================================================
# SHORT CONDITION
# ============================================================

def short_condition(df, i):

    row = df.iloc[i]

    prev = df.iloc[i - 1]

    # STRONG DOWNTREND
    trend = (
        row['EMA50']
        <
        row['EMA200'] * 0.998
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
    bearish = (
        row['close']
        <
        row['open']
    )

    # VOLUME
    volume = (
        row['volume']
        >
        row['VOL_SMA']
    )

    # VOLATILITY
    volatility = (
        row['ATR']
        >
        row['ATR_MEAN']
    )

    return (
        trend
        and
        rsi
        and
        stoch
        and
        bearish
        and
        volume
        and
        volatility
    )

# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest(df):

    print("Running backtest...")

    balance = STARTING_CAPITAL

    equity_curve = []

    trades = []

    in_trade = False

    trade_type = None

    entry_price = 0

    tp_price = 0

    sl_price = 0

    trade_capital = STARTING_CAPITAL

    for i in range(200, len(df)):

        row = df.iloc[i]

        # ====================================================
        # COMPOUNDING
        # ====================================================

        total_profit = (
            balance
            -
            STARTING_CAPITAL
        )

        trade_capital = (
            STARTING_CAPITAL
            +
            (total_profit * 0.5)
        )

        if trade_capital < STARTING_CAPITAL:
            trade_capital = STARTING_CAPITAL

        # ====================================================
        # ENTRY
        # ====================================================

        if not in_trade:

            # LONG
            if long_condition(df, i):

                in_trade = True

                trade_type = "LONG"

                entry_price = row['close']

                tp_move = (
                    TP_PERCENT
                    / LEVERAGE
                )

                sl_move = (
                    SL_PERCENT
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
                    "entry": entry_price,
                    "capital": trade_capital
                })

                print(
                    f"LONG ENTRY | "
                    f"Capital: ₹{round(trade_capital,2)}"
                )

            # SHORT
            elif short_condition(df, i):

                in_trade = True

                trade_type = "SHORT"

                entry_price = row['close']

                tp_move = (
                    TP_PERCENT
                    / LEVERAGE
                )

                sl_move = (
                    SL_PERCENT
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
                    "entry": entry_price,
                    "capital": trade_capital
                })

                print(
                    f"SHORT ENTRY | "
                    f"Capital: ₹{round(trade_capital,2)}"
                )

        # ====================================================
        # TRADE MANAGEMENT
        # ====================================================

        else:

            current_price = row['close']

            # LONG
            if trade_type == "LONG":

                # TAKE PROFIT
                if current_price >= tp_price:

                    gross_profit = (
                        trade_capital
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
                        f"+₹{round(net_profit,2)} | "
                        f"Balance: ₹{round(balance,2)}"
                    )

                    in_trade = False

                # STOP LOSS
                elif current_price <= sl_price:

                    gross_loss = (
                        trade_capital
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
                        f"-₹{round(total_loss,2)} | "
                        f"Balance: ₹{round(balance,2)}"
                    )

                    in_trade = False

            # SHORT
            elif trade_type == "SHORT":

                # TAKE PROFIT
                if current_price <= tp_price:

                    gross_profit = (
                        trade_capital
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
                        f"+₹{round(net_profit,2)} | "
                        f"Balance: ₹{round(balance,2)}"
                    )

                    in_trade = False

                # STOP LOSS
                elif current_price >= sl_price:

                    gross_loss = (
                        trade_capital
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
                        f"-₹{round(total_loss,2)} | "
                        f"Balance: ₹{round(balance,2)}"
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
        "ETH/USDT Advanced Futures Strategy"
    )

    plt.xlabel("Trades")

    plt.ylabel("Balance (₹)")

    plt.grid(True)

    plt.show()

# ============================================================
# MAIN
# ============================================================

print("====================================")
print("ETH/USDT ADVANCED FUTURES BACKTEST")
print("====================================")

df = get_binance_data()

df = add_indicators(df)

trades, equity_curve = backtest(df)

# ============================================================
# RESULTS
# ============================================================

trades_df = pd.DataFrame(trades)

# LONG STATS
long_trades = trades_df[
    trades_df['type'] == 'LONG'
]

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

long_profit = (
    long_trades['pnl'].sum()
)

# SHORT STATS
short_trades = trades_df[
    trades_df['type'] == 'SHORT'
]

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

short_profit = (
    short_trades['pnl'].sum()
)

# FINAL STATS
total_trades = len(trades_df)

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
    wins / total_trades
) * 100 if total_trades > 0 else 0

total_profit = (
    trades_df['pnl'].sum()
)

average_trade = (
    trades_df['pnl'].mean()
)

# ============================================================
# PRINT RESULTS
# ============================================================

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

print(
    f"Total Trades: "
    f"{total_trades}"
)

print(
    f"Wins: "
    f"{wins}"
)

print(
    f"Losses: "
    f"{losses}"
)

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
    f"₹{round(average_trade,2)}"
)

# ============================================================
# PLOT
# ============================================================

plot_results(equity_curve)