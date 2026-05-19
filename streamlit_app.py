import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="NIFTY Live Analyzer", layout="wide")

@dataclass
class SignalResult:
    signal: str
    strength: int
    score: float
    close: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target_1: float
    target_2: float
    reasons: list
    rsi: float
    atr: float
    sma20: float
    sma50: float
    volume: float
    vol_avg: float


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


@st.cache_data(ttl=300)
def fetch_data(period: str, interval: str) -> pd.DataFrame:
    df = yf.download("^NSEI", period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("No data returned for ^NSEI")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def compute_signal(df: pd.DataFrame) -> SignalResult:
    d = df.copy()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["EMA12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA26"] = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["RSI14"] = rsi(d["Close"], 14)
    d["ATR14"] = atr(d, 14)
    d["RET20"] = d["Close"].pct_change(20)
    d["VOL_MA20"] = d["Volume"].rolling(20).mean()
    d["VOL_RATIO"] = d["Volume"] / d["VOL_MA20"]

    last = d.iloc[-1]
    prev = d.iloc[-2]
    score = 0
    reasons = []

    if last["Close"] > last["SMA20"]:
        score += 1
        reasons.append("Close above 20-day average")
    else:
        score -= 1
        reasons.append("Close below 20-day average")

    if last["SMA20"] > last["SMA50"]:
        score += 1.5
        reasons.append("20-day average above 50-day average")
    else:
        score -= 1.5
        reasons.append("20-day average below 50-day average")

    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 1
        reasons.append("MACD above signal line")
    else:
        score -= 1
        reasons.append("MACD below signal line")

    if 50 <= last["RSI14"] <= 68:
        score += 1
        reasons.append("RSI in constructive bullish range")
    elif last["RSI14"] > 75:
        score -= 1
        reasons.append("RSI overbought")
    elif last["RSI14"] < 35:
        score += 0.5
        reasons.append("RSI oversold bounce zone")
    else:
        score -= 0.5
        reasons.append("RSI not supportive")

    if last["RET20"] > 0:
        score += 0.5
        reasons.append("20-session return positive")
    else:
        score -= 0.5
        reasons.append("20-session return negative")

    vol_ratio = float(last["VOL_RATIO"]) if pd.notna(last["VOL_RATIO"]) and np.isfinite(last["VOL_RATIO"]) else 1.0
    if vol_ratio > 1.1:
        if last["Close"] >= prev["Close"]:
            score += 0.75
            reasons.append("Above-average volume supports up move")
        else:
            score -= 0.75
            reasons.append("Above-average volume supports down move")
    else:
        reasons.append("Volume near average; conviction moderate")

    close = float(last["Close"])
    atrv = float(last["ATR14"]) if pd.notna(last["ATR14"]) else max(close * 0.01, 50)

    if score >= 2.5:
        signal = "BUY"
        entry_low = close - 0.25 * atrv
        entry_high = close + 0.10 * atrv
        stop_loss = close - 1.2 * atrv
        target_1 = close + 1.2 * atrv
        target_2 = close + 2.0 * atrv
    elif score <= -2.5:
        signal = "SELL"
        entry_low = close - 0.10 * atrv
        entry_high = close + 0.25 * atrv
        stop_loss = close + 1.2 * atrv
        target_1 = close - 1.2 * atrv
        target_2 = close - 2.0 * atrv
    else:
        signal = "HOLD"
        entry_low = close - 0.20 * atrv
        entry_high = close + 0.20 * atrv
        stop_loss = close - 1.0 * atrv
        target_1 = close + 1.0 * atrv
        target_2 = close + 1.6 * atrv

    strength = min(5, max(1, int(math.ceil(abs(score)))))
    return SignalResult(
        signal=signal,
        strength=strength,
        score=round(score, 2),
        close=round(close, 2),
        entry_low=round(entry_low, 2),
        entry_high=round(entry_high, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        reasons=reasons,
        rsi=round(float(last["RSI14"]), 2),
        atr=round(float(last["ATR14"]), 2),
        sma20=round(float(last["SMA20"]), 2),
        sma50=round(float(last["SMA50"]), 2),
        volume=float(last["Volume"]),
        vol_avg=round(float(last["VOL_MA20"]), 2) if pd.notna(last["VOL_MA20"]) else 0,
    )


def make_chart(df: pd.DataFrame) -> go.Figure:
    chart_df = df.copy()
    chart_df["SMA20"] = chart_df["Close"].rolling(20).mean()
    chart_df["SMA50"] = chart_df["Close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name="NIFTY",
    ))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50"], mode="lines", name="SMA50"))
    fig.update_layout(height=550, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


st.title("NIFTY 50 Live Analyzer")
st.caption("Live-style analysis using Yahoo Finance via yfinance. Educational use only, not financial advice.")

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)
    refresh = st.button("Refresh now")

if refresh:
    st.cache_data.clear()

try:
    df = fetch_data(period, interval)
    sig = compute_signal(df)
    last_date = df.index[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{sig.close:,.2f}")
    c2.metric("Volume", f"{sig.volume:,.0f}")
    c3.metric("RSI 14", f"{sig.rsi:.2f}")
    c4.metric("Score", f"{sig.score:.2f}")

    left, right = st.columns([1.7, 1])
    with left:
        st.plotly_chart(make_chart(df), use_container_width=True)
        st.dataframe(df.tail(20), use_container_width=True)

    with right:
        color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[sig.signal]
        st.markdown(f"## :{color}[{sig.signal}]")
        st.write(f"**As of:** {last_date}")
        st.write(f"**Strength:** {sig.strength}/5")
        st.write(f"**Entry zone:** {sig.entry_low:,.2f} to {sig.entry_high:,.2f}")
        st.write(f"**Stop loss:** {sig.stop_loss:,.2f}")
        st.write(f"**Target 1:** {sig.target_1:,.2f}")
        st.write(f"**Target 2:** {sig.target_2:,.2f}")
        st.write(f"**SMA20 / SMA50:** {sig.sma20:,.2f} / {sig.sma50:,.2f}")
        st.write(f"**ATR 14:** {sig.atr:,.2f}")
        st.write(f"**20D avg volume:** {sig.vol_avg:,.0f}")
        st.markdown("### Reasons")
        for r in sig.reasons:
            st.write(f"- {r}")
        st.info("Index volume can differ by provider, so treat volume as a supporting signal.")

except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()
