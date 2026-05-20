import math
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="NIFTY 50 Stock Scanner", layout="wide")

NIFTY50 = {
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "TCS.NS": "TCS",
    "BHARTIARTL.NS": "Bharti Airtel",
    "SBIN.NS": "State Bank of India",
    "LT.NS": "Larsen & Toubro",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharma",
    "M&M.NS": "Mahindra & Mahindra",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "NTPC.NS": "NTPC",
    "POWERGRID.NS": "Power Grid",
    "TITAN.NS": "Titan",
    "NESTLEIND.NS": "Nestle India",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "HCLTECH.NS": "HCL Technologies",
    "WIPRO.NS": "Wipro",
    "TECHM.NS": "Tech Mahindra",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel",
    "INDUSINDBK.NS": "IndusInd Bank",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "ONGC.NS": "ONGC",
    "COALINDIA.NS": "Coal India",
    "HDFCLIFE.NS": "HDFC Life",
    "SBILIFE.NS": "SBI Life",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "GRASIM.NS": "Grasim",
    "EICHERMOT.NS": "Eicher Motors",
    "DRREDDY.NS": "Dr Reddy's",
    "CIPLA.NS": "Cipla",
    "BRITANNIA.NS": "Britannia",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "BEL.NS": "Bharat Electronics",
    "TRENT.NS": "Trent",
    "BPCL.NS": "BPCL",
    "TATACONSUM.NS": "Tata Consumer",
    "LTIM.NS": "LTIMindtree",
}

@dataclass
class ScanRow:
    symbol: str
    name: str
    signal: str
    score: float
    strength: int
    close: float
    chg_pct: float
    volume_ratio: float
    rsi: float
    sma20: float
    sma50: float
    entry_low: float
    entry_high: float
    stop_loss: float
    target_1: float
    target_2: float
    reasons: List[str]
    last_candle_time: str
    stale_days: int
    freshness_flag: str


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
def fetch_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, ignore_tz=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def freshness_from_index(idx_value) -> tuple[str, int, str]:
    ts = pd.Timestamp(idx_value).to_pydatetime()
    now = datetime.utcnow()
    stale_days = max(0, (now.date() - ts.date()).days)
    ts_label = ts.strftime("%Y-%m-%d %H:%M")
    if stale_days >= 3:
        return ts_label, stale_days, "Stale"
    if stale_days >= 1:
        return ts_label, stale_days, "Check"
    return ts_label, stale_days, "Fresh"


def analyze_symbol(symbol: str, name: str) -> ScanRow | None:
    try:
        df = fetch_history(symbol)
        if len(df) < 60:
            return None
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
        d["CHG_PCT"] = d["Close"].pct_change() * 100

        last = d.iloc[-1]
        prev = d.iloc[-2]
        score = 0
        reasons = []

        if last["Close"] > last["SMA20"]:
            score += 1
            reasons.append("Above 20DMA")
        else:
            score -= 1
            reasons.append("Below 20DMA")

        if last["SMA20"] > last["SMA50"]:
            score += 1.5
            reasons.append("20DMA above 50DMA")
        else:
            score -= 1.5
            reasons.append("20DMA below 50DMA")

        if last["MACD"] > last["MACD_SIGNAL"]:
            score += 1
            reasons.append("MACD bullish")
        else:
            score -= 1
            reasons.append("MACD bearish")

        if 50 <= last["RSI14"] <= 68:
            score += 1
            reasons.append("RSI supportive")
        elif last["RSI14"] > 75:
            score -= 1
            reasons.append("RSI overbought")
        elif last["RSI14"] < 35:
            score += 0.5
            reasons.append("RSI oversold")
        else:
            score -= 0.5
            reasons.append("RSI mixed")

        if last["RET20"] > 0:
            score += 0.5
            reasons.append("20D return positive")
        else:
            score -= 0.5
            reasons.append("20D return negative")

        vol_ratio = float(last["VOL_RATIO"]) if pd.notna(last["VOL_RATIO"]) and np.isfinite(last["VOL_RATIO"]) else 1.0
        if vol_ratio > 1.1:
            if last["Close"] >= prev["Close"]:
                score += 0.75
                reasons.append("Volume supports up move")
            else:
                score -= 0.75
                reasons.append("Volume supports down move")
        else:
            reasons.append("Volume average")

        close = float(last["Close"])
        atrv = float(last["ATR14"])
        last_candle_time, stale_days, freshness_flag = freshness_from_index(d.index[-1])

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

        return ScanRow(
            symbol=symbol,
            name=name,
            signal=signal,
            score=round(score, 2),
            strength=min(5, max(1, int(math.ceil(abs(score))))),
            close=round(close, 2),
            chg_pct=round(float(last["CHG_PCT"]), 2) if pd.notna(last["CHG_PCT"]) else 0.0,
            volume_ratio=round(vol_ratio, 2),
            rsi=round(float(last["RSI14"]), 2),
            sma20=round(float(last["SMA20"]), 2),
            sma50=round(float(last["SMA50"]), 2),
            entry_low=round(entry_low, 2),
            entry_high=round(entry_high, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            reasons=reasons,
            last_candle_time=last_candle_time,
            stale_days=stale_days,
            freshness_flag=freshness_flag,
        )
    except Exception:
        return None


def make_chart(symbol: str) -> go.Figure:
    df = fetch_history(symbol, period="6mo", interval="1d")
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA50"))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


st.title("NIFTY 50 Stock Scanner")
st.caption("Scans NIFTY 50 stocks individually and ranks BUY / HOLD / SELL signals using Yahoo Finance data. Educational only.")
st.info("Prices come from Yahoo Finance via yfinance. Always cross-check execution price with NSE or your broker before trading.")

with st.sidebar:
    st.header("Scanner")
    top_n = st.slider("How many stocks to show", 5, 20, 10)
    signal_filter = st.selectbox("Signal filter", ["All", "BUY", "HOLD", "SELL"], index=1)
    refresh = st.button("Refresh scan")
    st.caption("Cache refresh interval: 5 minutes")

if refresh:
    st.cache_data.clear()
    st.toast("Cache cleared. Fresh scan started.")

scan_started = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
progress = st.progress(0, text="Scanning NIFTY 50 stocks...")
results: List[ScanRow] = []
items = list(NIFTY50.items())
for i, (symbol, name) in enumerate(items, start=1):
    row = analyze_symbol(symbol, name)
    if row is not None:
        results.append(row)
    progress.progress(i / len(items), text=f"Scanning {i}/{len(items)}: {name}")
progress.empty()

if not results:
    st.error("No stock data could be loaded right now.")
    st.stop()

scan_df = pd.DataFrame([
    {
        "Signal": r.signal,
        "Score": r.score,
        "Strength": r.strength,
        "Symbol": r.symbol,
        "Name": r.name,
        "Close": r.close,
        "1D %": r.chg_pct,
        "RSI": r.rsi,
        "Vol Ratio": r.volume_ratio,
        "SMA20": r.sma20,
        "SMA50": r.sma50,
        "Data Time": r.last_candle_time,
        "Age(D)": r.stale_days,
        "Status": r.freshness_flag,
        "Entry Low": r.entry_low,
        "Entry High": r.entry_high,
        "Stop Loss": r.stop_loss,
        "Target 1": r.target_1,
        "Target 2": r.target_2,
        "Reasons": ", ".join(r.reasons[:4]),
    }
    for r in results
]).sort_values(["Score", "RSI"], ascending=[False, False])

if signal_filter != "All":
    scan_df = scan_df[scan_df["Signal"] == signal_filter]

show_df = scan_df.head(top_n)

buy_count = int((pd.DataFrame([{ "Signal": r.signal } for r in results])["Signal"] == "BUY").sum())
hold_count = int((pd.DataFrame([{ "Signal": r.signal } for r in results])["Signal"] == "HOLD").sum())
sell_count = int((pd.DataFrame([{ "Signal": r.signal } for r in results])["Signal"] == "SELL").sum())
stale_count = int((pd.DataFrame([{ "stale": r.stale_days } for r in results])["stale"] >= 1).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BUY signals", buy_count)
c2.metric("HOLD signals", hold_count)
c3.metric("SELL signals", sell_count)
c4.metric("Stocks scanned", len(results))
c5.metric("Stale quotes", stale_count)

st.caption(f"Scan started at {scan_started}")
if stale_count > 0:
    st.warning("Some stocks are using older candles. Check the Data Time / Age(D) columns and verify price with NSE or your broker before trading.")

st.subheader("Scanner table")
st.dataframe(show_df, use_container_width=True, hide_index=True)

if len(show_df) > 0:
    selected_symbol = st.selectbox("Open chart for", show_df["Symbol"].tolist())
    picked = next(r for r in results if r.symbol == selected_symbol)
    left, right = st.columns([1.7, 1])
    with left:
        st.plotly_chart(make_chart(selected_symbol), use_container_width=True)
    with right:
        color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[picked.signal]
        st.markdown(f"## :{color}[{picked.signal}] {picked.symbol}")
        st.write(f"**Name:** {picked.name}")
        st.write(f"**Close:** {picked.close:,.2f}")
        st.write(f"**Data timestamp:** {picked.last_candle_time}")
        st.write(f"**Data age:** {picked.stale_days} day(s)")
        st.write(f"**Data status:** {picked.freshness_flag}")
        st.write(f"**Score:** {picked.score:.2f}")
        st.write(f"**Strength:** {picked.strength}/5")
        st.write(f"**RSI:** {picked.rsi:.2f}")
        st.write(f"**Volume ratio:** {picked.volume_ratio:.2f}")
        st.write(f"**Entry zone:** {picked.entry_low:,.2f} to {picked.entry_high:,.2f}")
        st.write(f"**Stop loss:** {picked.stop_loss:,.2f}")
        st.write(f"**Target 1:** {picked.target_1:,.2f}")
        st.write(f"**Target 2:** {picked.target_2:,.2f}")
        if picked.stale_days >= 1:
            st.warning("This price is not from the latest candle day. Cross-check on NSE or your broker before taking the trade.")
        st.markdown("### Reasons")
        for reason in picked.reasons:
            st.write(f"- {reason}")
        st.info("This is a rules-based technical scanner, not guaranteed financial advice.")
