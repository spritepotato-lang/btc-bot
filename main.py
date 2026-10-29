import os
import sys
import time

import pandas as pd
import requests

BASE_URL = "https://api.binance.us"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 1000
SL_ATR = 1.5
TP_ATR = 3.0
FEE = 0.001

TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")


def fetch_candles():
    r = requests.get(f"{BASE_URL}/api/v3/klines",
                     params={"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT},
                     timeout=15)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.iloc[:-1].reset_index(drop=True)  # closed candles only


def add_indicators(df):
    c = df["close"]
    df["ema_fast"] = c.ewm(span=20, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=50, adjust=False).mean()
    df["ema_trend"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    pc = c.shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    long_ok = ((df["ema_fast"] > df["ema_slow"]) & (c > df["ema_trend"])
               & (df["macd_hist"] > 0) & (df["macd_hist"] > df["macd_hist"].shift())
               & df["rsi"].between(50, 70))
    short_ok = ((df["ema_fast"] < df["ema_slow"]) & (c < df["ema_trend"])
                & (df["macd_hist"] < 0) & (df["macd_hist"] < df["macd_hist"].shift())
                & df["rsi"].between(30, 50))
    df["long"] = long_ok & ~long_ok.shift(fill_value=False)
    df["short"] = short_ok & ~short_ok.shift(fill_value=False)
    return df


def levels(row, side):
    p, a = row["close"], row["atr"]
    if side == "LONG":
        return p, p - SL_ATR * a, p + TP_ATR * a
    return p, p + SL_ATR * a, p - TP_ATR * a


def notify(msg):
    print(msg)
    if TG_TOKEN and TG_CHAT:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": msg}, timeout=15)
        r.raise_for_status()


def run_once():
    df = add_indicators(fetch_candles())
    row = df.iloc[-1]
    stamp = f"{row['time']:%Y-%m-%d %H:%M} UTC"
    # skip stale candles so a delayed run can't re-send an old signal
    age_min = (time.time() * 1000 - row["open_time"]) / 60000
    side = "LONG" if row["long"] else "SHORT" if row["short"] else None
    if side and age_min <= 30:
        entry, sl, tp = levels(row, side)
        notify(f"BTC 15m {side}\nCandle: {stamp}\nEntry ~{entry:,.0f}\n"
               f"Stop {sl:,.0f}\nTarget {tp:,.0f}\nRSI {row['rsi']:.0f}\n"
               f"(Signal only, not financial advice)")
    else:
        print(f"[{stamp}] no new signal (price {row['close']:,.0f}, RSI {row['rsi']:.0f})")


def backtest():
    df = add_indicators(fetch_candles())
    trades, i = [], 0
    while i < len(df) - 1:
        row = df.iloc[i]
        side = "LONG" if row["long"] else "SHORT" if row["short"] else None
        if not side:
            i += 1
            continue
        entry, sl, tp = levels(row, side)
        res, j = None, i + 1
        while j < len(df):
            hi, lo = df.at[j, "high"], df.at[j, "low"]
            hit_sl = lo <= sl if side == "LONG" else hi >= sl
            hit_tp = hi >= tp if side == "LONG" else lo <= tp
            if hit_sl:
                res = sl; break
            if hit_tp:
                res = tp; break
            j += 1
        if res is None:
            break
        ret = (res - entry) / entry if side == "LONG" else (entry - res) / entry
        trades.append(ret - 2 * FEE)
        i = j + 1
    if not trades:
        print("No completed trades in this sample.")
        return
    s = pd.Series(trades)
    print(f"Trades: {len(s)}  Win rate: {(s > 0).mean():.0%}  "
          f"Avg/trade: {s.mean():.3%}  Total: {(1 + s).prod() - 1:.1%}")


if __name__ == "__main__":
    backtest() if len(sys.argv) > 1 and sys.argv[1] == "backtest" else run_once()
