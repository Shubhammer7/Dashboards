
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import os
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Crypto Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BINANCE_BASE_URL = "https://api.binance.us"

API_KEY = os.environ.get("BINANCE_API_KEY", "")

COINS = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
}

INTERVALS = {
    "1 Hour": "1h",
    "4 Hours": "4h",
    "1 Day": "1d",
    "1 Week": "1w",
}

INTERVAL_LIMIT_MAP = {
    "1h": 500,
    "4h": 500,
    "1d": 365,
    "1w": 104,
}

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

        /* ── Root & Background ── */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #080c14 !important;
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1117 0%, #080c14 100%) !important;
            border-right: 1px solid #1e2d45;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        .block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 100% !important; }

        /* ── Typography ── */
        h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
        p, div, span, label { font-family: 'DM Mono', monospace !important; }

        /* ── KPI Cards ── */
        .kpi-card {
            background: linear-gradient(135deg, #0f1923 0%, #0d1521 100%);
            border: 1px solid #1e2d45;
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            position: relative;
            overflow: hidden;
            transition: border-color 0.3s ease, transform 0.2s ease;
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #00d4ff, #7b61ff);
            opacity: 0.7;
        }
        .kpi-card:hover {
            border-color: #00d4ff44;
            transform: translateY(-2px);
        }
        .kpi-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-family: 'Syne', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #f1f5f9;
            line-height: 1.1;
        }
        .kpi-sub {
            font-family: 'DM Mono', monospace;
            font-size: 0.72rem;
            margin-top: 0.35rem;
        }
        .kpi-positive { color: #22d3a0; }
        .kpi-negative { color: #f43f5e; }
        .kpi-neutral  { color: #94a3b8; }

        /* ── Section Headers ── */
        .section-header {
            font-family: 'Syne', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #cbd5e1;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.6rem 0 0.4rem 0;
            border-bottom: 1px solid #1e2d45;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: #00d4ff;
            display: inline-block;
        }

        /* ── Insight Cards ── */
        .insight-card {
            background: #0d1521;
            border: 1px solid #1e2d45;
            border-left: 3px solid #00d4ff;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.8rem;
            color: #94a3b8;
            transition: border-color 0.2s;
        }
        .insight-card:hover { border-left-color: #7b61ff; }
        .insight-card .badge {
            display: inline-block;
            font-size: 0.62rem;
            padding: 0.15rem 0.5rem;
            border-radius: 99px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .badge-bullish { background: #052e1c; color: #22d3a0; border: 1px solid #22d3a044; }
        .badge-bearish { background: #1f0d12; color: #f43f5e; border: 1px solid #f43f5e44; }
        .badge-neutral { background: #131b2a; color: #7b61ff; border: 1px solid #7b61ff44; }

        /* ── Status Bar ── */
        .status-bar {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            color: #64748b;
            padding: 0.4rem 0;
        }
        .status-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: #22d3a0;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* ── Disclaimer ── */
        .disclaimer {
            background: #0d1521;
            border: 1px solid #1e2d45;
            border-radius: 8px;
            padding: 0.7rem 1rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.72rem;
            color: #475569;
            text-align: center;
        }

        /* ── Streamlit Overrides ── */
        .stSelectbox > div > div { background: #0d1521 !important; border-color: #1e2d45 !important; }
        .stSlider > div { color: #94a3b8 !important; }
        div[data-testid="metric-container"] { display: none; }
        .stCheckbox label { font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; color: #94a3b8 !important; }
        .stDateInput input { background: #0d1521 !important; color: #e2e8f0 !important; border-color: #1e2d45 !important; }
        footer { display: none !important; }
        .stDownloadButton button {
            background: linear-gradient(135deg, #0f1923, #1a2744);
            border: 1px solid #1e2d45;
            color: #94a3b8;
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .stDownloadButton button:hover { border-color: #00d4ff; color: #00d4ff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV data from Binance Klines endpoint."""
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        raw = response.json()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df

    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch klines: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ticker_24h(symbol: str) -> dict:
    """Fetch 24h ticker statistics."""
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    try:
        r = requests.get(url, headers=headers, params={"symbol": symbol}, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch 24h ticker: {e}")
        return {}


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["rolling_vol"] = df["returns"].rolling(window).std()
    df["annualized_vol"] = df["rolling_vol"] * np.sqrt(252)
    return df


def compute_all_indicators(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    df = compute_moving_averages(df)
    df["RSI"] = compute_rsi(df["close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["close"])
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = compute_bollinger_bands(df["close"])
    df = compute_volatility(df, window=vol_window)
    return df

def linear_regression_forecast(series: pd.Series, horizon: int = 7):
    """Simple OLS linear regression for price forecasting."""
    from sklearn.linear_model import LinearRegression

    arr = series.dropna().values
    X = np.arange(len(arr)).reshape(-1, 1)
    y = arr

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(arr), len(arr) + horizon).reshape(-1, 1)
    forecast = model.predict(future_X)

    residuals = y - model.predict(X)
    std_err = residuals.std()

    return forecast, std_err

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,20,0.9)",
    font=dict(family="DM Mono, monospace", color="#94a3b8", size=11),
    xaxis=dict(
        gridcolor="#1e2d45",
        linecolor="#1e2d45",
        tickfont=dict(size=10),
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="#1e2d45",
        linecolor="#1e2d45",
        tickfont=dict(size=10),
        showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(13,21,33,0.8)",
        bordercolor="#1e2d45",
        borderwidth=1,
        font=dict(size=10),
    ),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
)


def apply_layout(fig, title: str = "", height: int = 420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=13, color="#cbd5e1")),
        height=height,
        **CHART_LAYOUT,
    )
    return fig

def render_candlestick_chart(df: pd.DataFrame, symbol: str, show_ma: bool, show_bb: bool):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name=symbol,
            increasing_line_color="#22d3a0",
            decreasing_line_color="#f43f5e",
            increasing_fillcolor="rgba(34,211,160,0.27)",
            decreasing_fillcolor="rgba(244,63,94,0.27)",
            line_width=1,
        ),
        row=1, col=1,
    )

    if show_ma and "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"],  name="MA20",  line=dict(color="#00d4ff", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"],  name="MA50",  line=dict(color="#7b61ff", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="#fbbf24", width=1.2, dash="dot")), row=1, col=1)

    if show_bb and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(color="rgba(148,163,184,0.27)", width=1), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(color="rgba(148,163,184,0.27)", width=1),
                                 fill="tonexty", fillcolor="rgba(148,163,184,0.04)", showlegend=True), row=1, col=1)

    # Volume bars
    colors = ["rgba(34,211,160,0.4)" if c >= o else "rgba(244,63,94,0.4)" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, showlegend=False),
        row=2, col=1,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,20,0.9)",
        font=dict(family="DM Mono, monospace", color="#94a3b8", size=11),
        height=520,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(13,21,33,0.8)", bordercolor="#1e2d45", borderwidth=1, font=dict(size=10)),
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout({ax: dict(gridcolor="#1e2d45", linecolor="#1e2d45", tickfont=dict(size=10))})

    return fig

def render_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(244,63,94,0.33)", line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(34,211,160,0.33)", line_width=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="#f43f5e", opacity=0.04, layer="below")
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#22d3a0", opacity=0.04, layer="below")

    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#7b61ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(123,97,255,0.06)",
    ))

    fig.add_annotation(x=df.index[-1], y=70, text="Overbought", showarrow=False,
                       xanchor="right", font=dict(color="#f43f5e", size=9))
    fig.add_annotation(x=df.index[-1], y=30, text="Oversold", showarrow=False,
                       xanchor="right", font=dict(color="#22d3a0", size=9))

    apply_layout(fig, "RSI (14)", height=240)
    fig.update_yaxes(range=[0, 100])
    return fig


def render_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#22d3a0" if v >= 0 else "#f43f5e" for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram", marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD",   line=dict(color="#00d4ff", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="#fbbf24", width=1.2, dash="dot")))
    apply_layout(fig, "MACD (12/26/9)", height=240)
    return fig

def render_volatility_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Rolling Volatility", "Returns Distribution"],
                        horizontal_spacing=0.08)

    fig.add_trace(go.Scatter(x=df.index, y=df["annualized_vol"] * 100,
                             fill="tozeroy", fillcolor="rgba(123,97,255,0.1)",
                             line=dict(color="#7b61ff", width=1.5), name="Ann. Vol %"), row=1, col=1)

    returns_clean = df["returns"].dropna() * 100
    fig.add_trace(go.Histogram(x=returns_clean, nbinsx=40,
                               marker_color="#00d4ff", opacity=0.7, name="Returns %"), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,12,20,0.9)",
        font=dict(family="DM Mono, monospace", color="#94a3b8", size=11),
        height=340, margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout({ax: dict(gridcolor="#1e2d45", linecolor="#1e2d45", tickfont=dict(size=10))})

    return fig

def render_correlation_section(df_btc: pd.DataFrame, df_eth: pd.DataFrame, vol_window: int):
    btc_returns = df_btc["close"].pct_change().rename("BTC")
    eth_returns = df_eth["close"].pct_change().rename("ETH")
    combined = pd.concat([btc_returns, eth_returns], axis=1).dropna()

    # Rolling correlation
    rolling_corr = combined["BTC"].rolling(vol_window).corr(combined["ETH"])

    col1, col2 = st.columns([1, 2])

    with col1:
        corr_matrix = combined.corr()
        fig_heat = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale=[[0, "#f43f5e"], [0.5, "#0d1521"], [1, "#22d3a0"]],
            zmin=-1, zmax=1,
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono, monospace", color="#94a3b8", size=11),
            height=280, margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False,
            title=dict(text="Correlation Matrix", font=dict(family="Syne, sans-serif", size=12, color="#cbd5e1")),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig_roll = go.Figure()
        fig_roll.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)
        fig_roll.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
            line=dict(color="#00d4ff", width=1.5), name=f"Rolling Corr ({vol_window})",
        ))
        apply_layout(fig_roll, f"Rolling Correlation (window={vol_window})", height=280)
        fig_roll.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig_roll, use_container_width=True, config={"displayModeBar": False})

    # Comparative returns
    fig_comp = go.Figure()
    btc_norm = (df_btc["close"] / df_btc["close"].iloc[0] - 1) * 100
    eth_norm = (df_eth["close"] / df_eth["close"].iloc[0] - 1) * 100
    fig_comp.add_trace(go.Scatter(x=df_btc.index, y=btc_norm, name="BTC %", line=dict(color="#f7931a", width=1.5)))
    fig_comp.add_trace(go.Scatter(x=df_eth.index, y=eth_norm, name="ETH %", line=dict(color="#627eea", width=1.5)))
    apply_layout(fig_comp, "Normalized Returns — BTC vs ETH (%)", height=300)
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    # Insight box
    latest_corr = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else 0
    if latest_corr > 0.7:
        corr_text = "BTC and ETH are highly correlated — they tend to move together, limiting diversification benefit during this period."
        badge_class = "badge-neutral"
    elif latest_corr > 0.3:
        corr_text = "Moderate correlation between BTC and ETH. Some diversification benefit exists, though assets share directional bias."
        badge_class = "badge-neutral"
    else:
        corr_text = "Low correlation detected — BTC and ETH are diverging. This presents a potential diversification opportunity."
        badge_class = "badge-bullish"

    st.markdown(f"""
    <div class="insight-card">
        <div class="badge {badge_class}">Correlation Insight</div><br>
        {corr_text}<br><br>
        <span style="color:#475569">Current rolling correlation: <strong style="color:#00d4ff">{latest_corr:.3f}</strong></span>
    </div>
    """, unsafe_allow_html=True)

def generate_insights(df: pd.DataFrame, symbol: str) -> list[dict]:
    insights = []
    if df.empty or len(df) < 30:
        return insights

    latest = df.iloc[-1]
    rsi = latest.get("RSI", np.nan)
    macd_val = latest.get("MACD", np.nan)
    macd_sig = latest.get("MACD_signal", np.nan)
    close = latest["close"]
    ann_vol = latest.get("annualized_vol", np.nan)
    ma20 = latest.get("MA20", np.nan)
    ma50 = latest.get("MA50", np.nan)
    bb_upper = latest.get("BB_upper", np.nan)
    bb_lower = latest.get("BB_lower", np.nan)

    if not np.isnan(rsi):
        if rsi > 70:
            insights.append({"badge": "badge-bearish", "label": "Overbought Signal",
                             "text": f"{symbol.replace('USDT','')} RSI at {rsi:.1f} — market is overbought. Watch for a potential pullback or consolidation phase."})
        elif rsi < 30:
            insights.append({"badge": "badge-bullish", "label": "Oversold Signal",
                             "text": f"{symbol.replace('USDT','')} RSI at {rsi:.1f} — deeply oversold territory. Potential mean reversion or bounce opportunity."})
        else:
            insights.append({"badge": "badge-neutral", "label": "RSI Neutral",
                             "text": f"{symbol.replace('USDT','')} RSI at {rsi:.1f} — neither overbought nor oversold. Momentum is balanced."})

    if not (np.isnan(macd_val) or np.isnan(macd_sig)):
        prev_macd = df["MACD"].iloc[-2] if len(df) > 2 else np.nan
        prev_sig  = df["MACD_signal"].iloc[-2] if len(df) > 2 else np.nan
        if macd_val > macd_sig and (np.isnan(prev_macd) or prev_macd <= prev_sig):
            insights.append({"badge": "badge-bullish", "label": "MACD Crossover",
                             "text": "Bullish MACD crossover detected — momentum turning positive. Historically a buy signal in trending markets."})
        elif macd_val < macd_sig and (np.isnan(prev_macd) or prev_macd >= prev_sig):
            insights.append({"badge": "badge-bearish", "label": "MACD Crossover",
                             "text": "Bearish MACD crossover detected — downward momentum building. Consider tightening stops or reducing exposure."})
        else:
            direction = "bullish" if macd_val > 0 else "bearish"
            insights.append({"badge": "badge-bullish" if macd_val > 0 else "badge-bearish", "label": "MACD Trend",
                             "text": f"MACD is {direction} at {macd_val:.2f}. No fresh crossover — current trend is in continuation mode."})

    if not (np.isnan(ma20) or np.isnan(ma50)):
        if ma20 > ma50:
            insights.append({"badge": "badge-bullish", "label": "Golden Cross Zone",
                             "text": f"MA20 ({ma20:,.0f}) is above MA50 ({ma50:,.0f}) — price structure remains bullish on this timeframe."})
        else:
            insights.append({"badge": "badge-bearish", "label": "Death Cross Zone",
                             "text": f"MA20 ({ma20:,.0f}) is below MA50 ({ma50:,.0f}) — bearish structure. Rallies may face resistance at moving averages."})

    if not (np.isnan(bb_upper) or np.isnan(bb_lower)):
        bb_width = (bb_upper - bb_lower) / close * 100
        if close > bb_upper:
            insights.append({"badge": "badge-bearish", "label": "Bollinger Band Breakout",
                             "text": f"Price has broken above the upper Bollinger Band — extended move. Expect volatility or mean reversion soon."})
        elif close < bb_lower:
            insights.append({"badge": "badge-bullish", "label": "Bollinger Band Squeeze",
                             "text": f"Price is below the lower Bollinger Band — potential oversold extreme. High-risk contrarian entry zone."})
        if bb_width < 5:
            insights.append({"badge": "badge-neutral", "label": "Volatility Squeeze",
                             "text": f"Bollinger Band width at {bb_width:.1f}% — historically low volatility. A significant directional move may be imminent."})

    if not np.isnan(ann_vol):
        vol_pct = ann_vol * 100
        if vol_pct > 80:
            insights.append({"badge": "badge-bearish", "label": "High Volatility Regime",
                             "text": f"Annualized volatility at {vol_pct:.1f}% — elevated risk environment. Position sizing should reflect heightened uncertainty."})
        elif vol_pct < 30:
            insights.append({"badge": "badge-neutral", "label": "Low Volatility",
                             "text": f"Annualized volatility at {vol_pct:.1f}% — unusually calm. Low-volatility periods often precede sharp moves."})

    return insights

def render_forecast_chart(df: pd.DataFrame, symbol: str, horizon: int = 7):
    forecast, std_err = linear_regression_forecast(df["close"], horizon)
    last_date = df.index[-1]
    freq_delta = df.index[-1] - df.index[-2]
    future_dates = [last_date + freq_delta * i for i in range(1, horizon + 1)]

    fig = go.Figure()

    # Historical
    hist_window = min(60, len(df))
    fig.add_trace(go.Scatter(
        x=df.index[-hist_window:], y=df["close"].iloc[-hist_window:],
        name="Historical", line=dict(color="#94a3b8", width=1.5),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast,
        name="Forecast", line=dict(color="#00d4ff", width=2, dash="dash"),
    ))

    # Confidence interval
    upper_ci = forecast + 1.96 * std_err
    lower_ci = forecast - 1.96 * std_err
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper_ci) + list(lower_ci[::-1]),
        fill="toself", fillcolor="rgba(0,212,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
    ))

    apply_layout(fig, f"{symbol} — Linear Regression Forecast (+{horizon} periods)", height=360)
    return fig

def render_kpi_cards(df: pd.DataFrame, ticker: dict, symbol: str):
    latest = df.iloc[-1] if not df.empty else {}

    price = float(ticker.get("lastPrice", latest.get("close", 0)))
    change_pct = float(ticker.get("priceChangePercent", 0))
    volume = float(ticker.get("quoteVolume", latest.get("quote_volume", 0)))
    ann_vol = latest.get("annualized_vol", 0) * 100 if not df.empty else 0
    rsi_val = latest.get("RSI", 0) if not df.empty else 0
    macd_val = latest.get("MACD", 0) if not df.empty else 0
    macd_sig = latest.get("MACD_signal", 0) if not df.empty else 0

    change_cls = "kpi-positive" if change_pct >= 0 else "kpi-negative"
    change_arrow = "+" if change_pct >= 0 else "-"
    rsi_cls = "kpi-negative" if rsi_val > 70 else ("kpi-positive" if rsi_val < 30 else "kpi-neutral")
    macd_cls = "kpi-positive" if macd_val > macd_sig else "kpi-negative"
    macd_signal_text = "Bullish" if macd_val > macd_sig else "Bearish"
    vol_cls = "kpi-negative" if ann_vol > 80 else ("kpi-positive" if ann_vol < 30 else "kpi-neutral")

    def fmt_volume(v):
        if v >= 1e9: return f"${v/1e9:.2f}B"
        if v >= 1e6: return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"

    cards = [
        {
            "label": "Current Price",
            "value": f"${price:,.2f}",
            "sub": f'<span class="{change_cls}">{change_arrow} {abs(change_pct):.2f}% (24h)</span>',
        },
        {
            "label": "24h Change",
            "value": f'{change_arrow} {abs(change_pct):.2f}%',
            "sub": f'<span class="kpi-neutral">vs yesterday close</span>',
            "cls": change_cls,
        },
        {
            "label": "24h Volume",
            "value": fmt_volume(volume),
            "sub": '<span class="kpi-neutral">Quote asset (USDT)</span>',
        },
        {
            "label": "Ann. Volatility",
            "value": f"{ann_vol:.1f}%",
            "sub": f'<span class="{vol_cls}">{"High" if ann_vol > 80 else "Low" if ann_vol < 30 else "Moderate"} regime</span>',
        },
        {
            "label": "RSI (14)",
            "value": f"{rsi_val:.1f}",
            "sub": f'<span class="{rsi_cls}">{"Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"}</span>',
        },
        {
            "label": "MACD Signal",
            "value": macd_signal_text,
            "sub": f'<span class="{macd_cls}">MACD: {macd_val:.2f} | Sig: {macd_sig:.2f}</span>',
            "cls": macd_cls,
        },
    ]

    cols = st.columns(6)
    for col, card in zip(cols, cards):
        val_cls = card.get("cls", "")
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{card['label']}</div>
                <div class="kpi-value {val_cls}">{card['value']}</div>
                <div class="kpi-sub">{card['sub']}</div>
            </div>
            """, unsafe_allow_html=True)

def main():
    inject_css()
    col_logo, col_status = st.columns([3, 1])
    with col_logo:
        st.markdown("""
        <h1 style="font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;
                   background:linear-gradient(90deg,#00d4ff,#7b61ff);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:0;padding:0;">
            Crypto Analytics Dashboard
        </h1>
        <p style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#475569;margin:0.2rem 0 1rem 0;">
            Institutional-grade cryptocurrency analytics · Powered by Binance
        </p>
        """, unsafe_allow_html=True)
    with col_status:
        ts = datetime.utcnow().strftime("%H:%M:%S UTC")
        st.markdown(f"""
        <div class="status-bar" style="justify-content:flex-end;margin-top:0.8rem;">
            <div class="status-dot"></div>
            <span>Live · {ts}</span>
        </div>
        """, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;color:#cbd5e1;padding:0.5rem 0 1rem 0;">Dashboard Controls</div>', unsafe_allow_html=True)

        coin_label = st.selectbox("Coin", list(COINS.keys()), index=0)
        symbol = COINS[coin_label]

        interval_label = st.selectbox("Interval", list(INTERVALS.keys()), index=2)
        interval = INTERVALS[interval_label]

        st.markdown("---")
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Indicator Toggles</div>', unsafe_allow_html=True)
        show_ma  = st.checkbox("Moving Averages (MA20/50, EMA20)", value=True)
        show_bb  = st.checkbox("Bollinger Bands", value=True)
        show_rsi = st.checkbox("RSI Indicator", value=True)
        show_macd= st.checkbox("MACD", value=True)

        st.markdown("---")
        vol_window = st.slider("Volatility Window", min_value=7, max_value=60, value=20, step=1)
        show_corr  = st.checkbox("Correlation Analysis (BTC vs ETH)", value=True)
        show_forecast = st.checkbox("ML Forecast", value=True)

        st.markdown("---")
        auto_refresh = st.checkbox("Auto Refresh (5 min)", value=False)
        if auto_refresh:
            try:
                st.cache_data.clear()
            except Exception:
                pass
            time.sleep(1)
            try:
                st.experimental_rerun()
            except Exception:
                st.stop()

    limit = INTERVAL_LIMIT_MAP.get(interval, 365)

    with st.spinner("Fetching market data from Binance…"):
        df_raw = fetch_klines(symbol, interval, limit)
        ticker = fetch_ticker_24h(symbol)

    if df_raw.empty:
        st.error("Failed to fetch data. Please check your API connection or try again later.")
        st.stop()

    df = compute_all_indicators(df_raw, vol_window=vol_window)
    render_kpi_cards(df, ticker, symbol)
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header"><span class="section-dot"></span>Price Analytics</div>', unsafe_allow_html=True)
    fig_candle = render_candlestick_chart(df, symbol, show_ma, show_bb)
    st.plotly_chart(fig_candle, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["autoScale2d"]})

    st.markdown('<div class="section-header"><span class="section-dot"></span>Technical Indicators</div>', unsafe_allow_html=True)
    col_rsi, col_macd = st.columns(2)
    if show_rsi:
        with col_rsi:
            st.plotly_chart(render_rsi_chart(df), use_container_width=True, config={"displayModeBar": False})
    if show_macd:
        with col_macd:
            st.plotly_chart(render_macd_chart(df), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-header"><span class="section-dot"></span>Volatility Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(render_volatility_chart(df), use_container_width=True, config={"displayModeBar": False})

    latest_vol = df["annualized_vol"].dropna().iloc[-1] * 100 if not df["annualized_vol"].dropna().empty else 0
    prev_vol   = df["annualized_vol"].dropna().iloc[-8] * 100 if len(df["annualized_vol"].dropna()) > 8 else latest_vol
    vol_change = latest_vol - prev_vol
    vol_direction = "increased" if vol_change > 0 else "decreased"

    st.markdown(f"""
    <div class="insight-card">
        <div class="badge badge-neutral">Volatility Insight</div><br>
        Annualized volatility has <strong style="color:#00d4ff">{vol_direction}</strong> by 
        <strong style="color:#{'f43f5e' if vol_change > 0 else '22d3a0'}">{abs(vol_change):.1f}pp</strong> 
        over the last 8 periods. Current annualized volatility stands at 
        <strong style="color:#00d4ff">{latest_vol:.1f}%</strong>
        — {"elevated risk environment requiring careful position sizing." if latest_vol > 60 else "moderate volatility environment suitable for systematic strategies."}
    </div>
    """, unsafe_allow_html=True)

    if show_corr:
        st.markdown('<div class="section-header"><span class="section-dot"></span>Correlation Analysis — BTC vs ETH</div>', unsafe_allow_html=True)
        with st.spinner("Fetching ETH data for correlation…"):
            df_btc_raw = fetch_klines("BTCUSDT", interval, limit)
            df_eth_raw = fetch_klines("ETHUSDT", interval, limit)

        if not df_btc_raw.empty and not df_eth_raw.empty:
            render_correlation_section(df_btc_raw, df_eth_raw, vol_window)

    st.markdown('<div class="section-header"><span class="section-dot"></span>Market Insights</div>', unsafe_allow_html=True)
    insights = generate_insights(df, symbol)
    if insights:
        insight_cols = st.columns(2)
        for i, ins in enumerate(insights):
            with insight_cols[i % 2]:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="badge {ins['badge']}">{ins['label']}</div><br>
                    {ins['text']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Insufficient data to generate insights. Try a longer timeframe.")

    if show_forecast:
        st.markdown('<div class="section-header"><span class="section-dot"></span>ML Price Forecast</div>', unsafe_allow_html=True)
        fig_fc = render_forecast_chart(df, symbol, horizon=7)
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})
        st.markdown("""
        <div class="disclaimer">
            This forecast is generated using Linear Regression on historical price data. 
            It is for <strong>educational and analytical purposes only</strong> and does not constitute financial advice. 
            Past performance is not indicative of future results.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    export_cols = st.columns([1, 1, 4])
    with export_cols[0]:
        csv = df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export OHLCV + Indicators",
            data=csv,
            file_name=f"{symbol}_{interval}_crypto_analytics.csv",
            mime="text/csv",
        )
    with export_cols[1]:
        ticker_df = pd.DataFrame([ticker])
        t_csv = ticker_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export 24h Ticker",
            data=t_csv,
            file_name=f"{symbol}_ticker.csv",
            mime="text/csv",
        )

    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem;
                font-family:'DM Mono',monospace;font-size:0.65rem;color:#334155;">
        Crypto Analytics Dashboard · Data from Binance Public API · Not financial advice
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
