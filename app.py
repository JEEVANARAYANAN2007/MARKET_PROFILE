# app.py
# Minimal Streamlit app: 30m candlesticks + Market Profile (TPO)
# Works in DEMO mode without any API keys (synthetic data).
# Later you can enable Kite/Upstox by adding real fetchers (instructions below).

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import math
import random
from collections import defaultdict

st.set_page_config(layout="wide", page_title="Market Profile Demo")

# -------------------------
# Helper: generate synthetic intraday minute data for demo
# -------------------------
def synthetic_intraday(symbol, days=1):
    # Trading day times: 09:15 to 15:30
    minutes_per_day = (15*60 + 30) - (9*60 + 15)  # 375 minutes
    all_rows = []
    base = 1000 + (sum(ord(c) for c in symbol) % 2000)  # deterministic-ish base per symbol
    for d in range(days):
        date = (dt.date.today() - dt.timedelta(days=(days - 1 - d)))
        # produce random walk starting near base
        price = base + random.uniform(-5, 5)
        for m in range(minutes_per_day):
            t = dt.datetime.combine(date, dt.time(9, 15)) + dt.timedelta(minutes=m)
            # small random walk
            price *= (1 + random.uniform(-0.0008, 0.0008))
            # occasional slightly larger change
            if random.random() < 0.002:
                price *= (1 + random.uniform(-0.01, 0.01))
            volume = int(abs(np.random.normal(2000, 800)))
            all_rows.append({"datetime": t, "close": round(price, 2), "open": round(price*(1+random.uniform(-0.001,0.001)),2),
                             "high": round(price*(1+random.uniform(0,0.002)),2), "low": round(price*(1-random.uniform(0,0.002)),2),
                             "volume": volume})
    df = pd.DataFrame(all_rows).set_index("datetime")
    return df

# -------------------------
# TPO builder (per your spec)
# -------------------------
def build_tpo_from_minute_df(df_min, interval_minutes=30, tick_size=0.5, days=1):
    """
    Input: df_min with DatetimeIndex of minute samples and 'close' column.
    Steps:
      - Split into day groups (last `days` days)
      - For each day, create slices every interval_minutes starting at 09:15
      - Each slice gets a letter (A, B, C ...)
      - For each minute in slice, round price to nearest tick and record letter
    Returns:
      dict with bins (sorted desc), poc, vah, val, total_tpos
    """
    if df_min.empty:
        return {"bins": [], "poc": None, "vah": None, "val": None, "total_tpos": 0}

    df = df_min.sort_index()
    # group by date
    grouped = [g for _, g in df.groupby(df.index.date)]
    grouped = grouped[-days:]  # last `days` days
    bins_map = defaultdict(list)  # price -> list of letters (chronological)
    total_tpos = 0

    for day_df in grouped:
        if day_df.empty:
            continue
        # reference start at 09:15 for that date
        date = day_df.index[0].date()
        ref_start = dt.datetime.combine(date, dt.time(9,15))
        last_ts = day_df.index[-1]
        slice_start = ref_start
        slice_index = 0
        while slice_start <= last_ts:
            slice_end = slice_start + pd.Timedelta(minutes=interval_minutes)
            letter = chr(ord('A') + (slice_index % 26))
            mask = (day_df.index >= slice_start) & (day_df.index < slice_end)
            slice_prices = day_df.loc[mask, "close"].values if not day_df.loc[mask].empty else []
            for p in slice_prices:
                lvl = round(p / tick_size) * tick_size
                bins_map[round(lvl, 2)].append({"letter": letter, "time": slice_start.isoformat()})
                total_tpos += 1
            slice_index += 1
            slice_start = slice_end

    # convert bins_map to bins list sorted descending price
    sorted_prices = sorted(bins_map.keys(), reverse=True)
    bins = []
    for price in sorted_prices:
        entries = bins_map[price]
        # aggregate by letter
        grouped_letters = {}
        for e in entries:
            L = e["letter"]
            grouped_letters.setdefault(L, {"letter": L, "count": 0, "times": []})
            grouped_letters[L]["count"] += 1
            grouped_letters[L]["times"].append(e["time"])
        tpos = [ {"letter": v["letter"], "count": v["count"], "times": v["times"]} for k,v in sorted(grouped_letters.items()) ]
        bins.append({"price": float(price), "tpos": tpos, "total": sum([t["count"] for t in tpos])})

    # compute POC & Value Area (70%) using greedy expansion from POC
    total = sum(b["total"] for b in bins) or 1
    poc = None; vah = None; val = None
    if total > 0:
        poc_bin = max(bins, key=lambda b: b["total"])
        poc = poc_bin["price"]
        price_totals = sorted([(b["price"], b["total"]) for b in bins], key=lambda x: x[0])
        prices_only = [p for p,_ in price_totals]
        nearest_idx = min(range(len(prices_only)), key=lambda i: abs(prices_only[i]-poc))
        target = total * 0.7
        cum = price_totals[nearest_idx][1]
        low_idx = high_idx = nearest_idx
        while cum < target:
            left_val = price_totals[low_idx-1][1] if low_idx-1 >= 0 else -1
            right_val = price_totals[high_idx+1][1] if high_idx+1 < len(price_totals) else -1
            if left_val >= right_val and left_val != -1:
                low_idx -= 1
                cum += price_totals[low_idx][1]
            elif right_val != -1:
                high_idx += 1
                cum += price_totals[high_idx][1]
            else:
                break
        val = price_totals[low_idx][0]
        vah = price_totals[high_idx][0]

    return {"bins": bins, "poc": poc, "vah": vah, "val": val, "total_tpos": total}

# -------------------------
# (Optional) Placeholders for real broker fetchers
# -------------------------
def fetch_real_from_kite(symbol, interval="30m", days=1):
    # This function is a placeholder. To enable real data:
    # 1) install kiteconnect: pip install kiteconnect
    # 2) use KiteConnect with your API key & access token
    # 3) convert to minute dataframe similar to synthetic_intraday
    raise NotImplementedError("Kite fetcher not implemented in demo. Use DEMO mode to test app.")

def fetch_real_from_upstox(symbol, interval="30m", days=1):
    # Placeholder for Upstox or other broker
    raise NotImplementedError("Upstox fetcher not implemented in demo. Use DEMO mode to test app.")

# -------------------------
# Streamlit UI
# -------------------------
st.title("Market Profile (TPO) — Demo (30m)")

with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("NSE Symbol (no suffix)", value="RELIANCE")
    days = st.selectbox("Days", options=[1,3,5,10], index=0)
    interval_minutes = st.selectbox("Interval (minutes)", options=[30], index=0)  # fixed 30 for now
    tick_size = st.number_input("Price tick (rounding)", value=0.5, step=0.1)
    mode = st.selectbox("Data mode", options=["DEMO (no API)", "Zerodha (Kite) - requires keys", "Upstox - requires keys"])
    run_live = st.checkbox("Live auto-refresh (demo data)", value=False)
    refresh_sec = st.selectbox("Refresh every (sec)", options=[10,30,60], index=1)

    st.markdown("---")
    st.markdown("**Kite / Upstox keys (optional)**")
    kite_api_key = st.text_input("Kite API Key (optional)", type="password")
    kite_access_token = st.text_input("Kite Access Token (optional)", type="password")
    # Note: we don't store keys anywhere in demo; Streamlit Secrets is recommended for production

# Info row
if mode != "DEMO (no API)":
    if (not kite_api_key) or (not kite_access_token):
        st.warning("You selected a broker mode but did not provide keys. The app will run in DEMO mode.")
        effective_mode = "DEMO"
    else:
        st.info("Broker keys provided. (Note: real broker integration not implemented in this demo file.)")
        effective_mode = "BROKER"
else:
    effective_mode = "DEMO"

# Fetch data (demo or real)
@st.cache_data(ttl=30)
def load_data(symbol, days, mode):
    if mode == "DEMO":
        df_min = synthetic_intraday(symbol, days=days)
    else:
        # here you would call real fetcher; demo falls back to synthetic
        try:
            df_min = synthetic_intraday(symbol, days=days)
        except Exception:
            df_min = synthetic_intraday(symbol, days=days)
    return df_min

df_min = load_data(symbol, days, effective_mode)

# Build 30m candles from minute series (for plotting)
def build_candles_from_minute(df_min, interval_minutes=30):
    if df_min.empty:
        return pd.DataFrame()
    # We will resample using fixed 30-minute windows aligned from 09:15:
    # create a 'slice_id' column: minutes since 09:15 floored to interval
    def slice_id_for_ts(ts):
        date = ts.date()
        ref = dt.datetime.combine(date, dt.time(9,15))
        minutes = int((ts - ref).total_seconds() // 60)
        if minutes < 0:
            # before market open: put in first bucket
            return 0
        return minutes // interval_minutes

    df = df_min.copy()
    df = df.reset_index()
    df['slice_id'] = df['datetime'].apply(slice_id_for_ts)
    # aggregate per slice_id and date
    df['date'] = df['datetime'].dt.date
    grouped = df.groupby(['date','slice_id'])
    rows = []
    for (date, sid), g in grouped:
        # slice start time:
        ref = dt.datetime.combine(date, dt.time(9,15)) + dt.timedelta(minutes=sid*interval_minutes)
        open_ = g.iloc[0]['open']
        close = g.iloc[-1]['close']
        high = g['high'].max()
        low = g['low'].min()
        vol = int(g['volume'].sum())
        rows.append({"datetime": ref, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
    candle_df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return candle_df

candle_df = build_candles_from_minute(df_min, interval_minutes=interval_minutes)

# Build TPO from minute-level data
tpo_res = build_tpo_from_minute_df(df_min, interval_minutes=interval_minutes, tick_size=tick_size, days=days)

# Layout: two columns
col1, col2 = st.columns([3,1])

with col1:
    st.subheader(f"{symbol} — {interval_minutes}m Candles")
    if candle_df.empty:
        st.info("No candle data available.")
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=candle_df.index,
            open=candle_df['open'],
            high=candle_df['high'],
            low=candle_df['low'],
            close=candle_df['close'],
            increasing_line_color='green', decreasing_line_color='red',
            showlegend=False
        )])
        # add POC/VA lines across the candle chart
        if tpo_res.get("poc") is not None:
            fig.add_hline(y=tpo_res["poc"], line=dict(color="yellow", dash="dash"), annotation_text="POC", annotation_position="top left")
        if tpo_res.get("vah") is not None and tpo_res.get("val") is not None:
            fig.add_hline(y=tpo_res["vah"], line=dict(color="orange", dash="dot"), annotation_text="VAH", annotation_position="top left")
            fig.add_hline(y=tpo_res["val"], line=dict(color="orange", dash="dot"), annotation_text="VAL", annotation_position="bottom left")
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=480, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Market Profile (TPO)")
    # show data source badge
    if effective_mode == "DEMO":
        st.markdown("**Mode:** 🔶 DEMO (synthetic / delayed data)")
    else:
        st.markdown("**Mode:** 🟢 LIVE broker (not enabled in demo)")

    # Show POC / VA
    st.markdown(f"**POC:** {tpo_res.get('poc')}  &nbsp;&nbsp; **VAH:** {tpo_res.get('vah')}  &nbsp;&nbsp; **VAL:** {tpo_res.get('val')}")
    bins = tpo_res.get("bins", [])
    if not bins:
        st.info("No TPO bins generated (no data).")
    else:
        # simple horizontal bar chart (price vs total TPO count)
        prices = [b['price'] for b in bins][::-1]  # ascending for human reading
        counts = [b['total'] for b in bins][::-1]
        hover = []
        for b in bins[::-1]:
            # create a compact letters string for hover
            letters_str = " ".join([f"{tp['letter']}({tp['count']})" for tp in b['tpos']])
            hover.append(f"price: {b['price']}<br>letters: {letters_str}")
        prof_fig = go.Figure(go.Bar(x=counts, y=[str(p) for p in prices], orientation='h', text=counts, hovertext=hover, hoverinfo="text"))
        prof_fig.update_layout(height=480, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10), yaxis={'categoryorder':'array','categoryarray': [str(p) for p in prices[::-1]]})
        st.plotly_chart(prof_fig, use_container_width=True)

# Footer / instructions
st.markdown("---")
st.markdown("**Notes:** This is demo mode with synthetic intraday data so you can test the app before getting any API keys. "
            "To enable real live NSE data, obtain Zerodha (Kite) or Upstox API credentials and replace the demo fetcher with real fetcher code. "
            "If you want, I can provide the exact snippet to plug in Kite Connect or Upstox next.")
