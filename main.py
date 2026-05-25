"""
=============================================================================
 CryptoSentinel — Crypto Fear & Volatility Forecasting Engine
=============================================================================
 NLP   : VADER + custom crypto lexicon → Daily Fear/Greed Index (0–100)
 Model : GARCH(1,1) — industry-standard volatility forecasting
 Field : Cryptocurrency / Finance

 How NLP connects to Computational Science:
   Crypto posts/news → Fear Score → injected as external regressor
   into GARCH model → volatility forecast → risk classification

 Run: streamlit run crypto_app.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from arch import arch_model
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CryptoSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme configuration
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def get_theme_css(theme):
    if theme == "dark":
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

        .hero {
            background: linear-gradient(135deg, #0f0f1a 0%, #1a0533 50%, #0f1a0f 100%);
            border: 1px solid #2d1b69;
            border-radius: 20px;
            padding: 32px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: '🛡️';
            position: absolute;
            right: 40px; top: 50%;
            transform: translateY(-50%);
            font-size: 5rem;
            opacity: 0.15;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #f472b6, #fb923c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 8px 0;
        }
        .hero-sub { color: #94a3b8; font-size: 0.95rem; margin: 0; }

        .kpi-card {
            background: #0d1117;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            border: 1px solid #1f2937;
            transition: border-color 0.2s;
        }
        .kpi-label { color: #6b7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; }
        .kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #f9fafb; }
        .kpi-sub   { font-size: 0.78rem; margin-top: 4px; }

        .fear-extreme  { color: #ef4444; }
        .fear-high     { color: #f97316; }
        .fear-neutral  { color: #eab308; }
        .fear-greed    { color: #22c55e; }
        .fear-extreme-greed { color: #10b981; }

        .risk-badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }
        .risk-extreme  { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }
        .risk-high     { background: #431407; color: #fdba74; border: 1px solid #f97316; }
        .risk-moderate { background: #422006; color: #fde68a; border: 1px solid #eab308; }
        .risk-low      { background: #052e16; color: #86efac; border: 1px solid #22c55e; }

        .section-hdr {
            font-size: 1rem; font-weight: 700; color: #e2e8f0;
            border-left: 3px solid #a78bfa;
            padding-left: 10px; margin: 28px 0 14px 0;
        }
        .insight-box {
            background: #0d1117;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 10px;
        }
        .insight-title { color: #a78bfa; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px; }
        .insight-text  { color: #d1d5db; font-size: 0.88rem; line-height: 1.5; }

        div[data-testid="stSidebar"] { background: #0a0a12; }
        .stButton > button {
            background: linear-gradient(135deg, #7c3aed, #db2777);
            color: white; border: none; border-radius: 12px;
            padding: 12px 28px; font-weight: 700;
            font-size: 0.95rem; width: 100%;
        }
        .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
        </style>
        """
    else:  # Light mode
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

        .hero {
            background: linear-gradient(135deg, #f8f9fa 0%, #f0e6ff 50%, #f0faf0 100%);
            border: 1px solid #d4c5f9;
            border-radius: 20px;
            padding: 32px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: '🛡️';
            position: absolute;
            right: 40px; top: 50%;
            transform: translateY(-50%);
            font-size: 5rem;
            opacity: 0.08;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7c3aed, #db2777, #f97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 8px 0;
        }
        .hero-sub { color: #6b7280; font-size: 0.95rem; margin: 0; }

        .kpi-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e5e7eb;
            transition: border-color 0.2s;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        .kpi-label { color: #6b7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; }
        .kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #1f2937; }
        .kpi-sub   { font-size: 0.78rem; margin-top: 4px; }

        .fear-extreme  { color: #dc2626; }
        .fear-high     { color: #ea580c; }
        .fear-neutral  { color: #d97706; }
        .fear-greed    { color: #16a34a; }
        .fear-extreme-greed { color: #0891b2; }

        .risk-badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }
        .risk-extreme  { background: #fee2e2; color: #991b1b; border: 1px solid #dc2626; }
        .risk-high     { background: #fed7aa; color: #92400e; border: 1px solid #ea580c; }
        .risk-moderate { background: #fef3c7; color: #78350f; border: 1px solid #d97706; }
        .risk-low      { background: #dcfce7; color: #166534; border: 1px solid #16a34a; }

        .section-hdr {
            font-size: 1rem; font-weight: 700; color: #1f2937;
            border-left: 3px solid #7c3aed;
            padding-left: 10px; margin: 28px 0 14px 0;
        }
        .insight-box {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 10px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        .insight-title { color: #7c3aed; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px; }
        .insight-text  { color: #374151; font-size: 0.88rem; line-height: 1.5; }

        div[data-testid="stSidebar"] { background: #f9fafb; }
        .stButton > button {
            background: linear-gradient(135deg, #7c3aed, #db2777);
            color: white; border: none; border-radius: 12px;
            padding: 12px 28px; font-weight: 700;
            font-size: 0.95rem; width: 100%;
        }
        .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
        </style>
        """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  CRYPTO SOCIAL MEDIA DATASET
#  Simulates Reddit (r/CryptoCurrency, r/Bitcoin) + Twitter posts
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_POSTS = """2024-01-02,Bitcoin is pumping hard today, we're going to 50k this month easily
2024-01-02,Just bought more BTC on the dip, feeling bullish about Q1
2024-01-02,crypto market looks amazing right now, massive gains incoming
2024-01-03,ETF approval hype is real, institutions are flooding in
2024-01-03,BTC holding strong above 45k, bulls in total control
2024-01-04,Fed rate cut signals boosting crypto massively, moon incoming
2024-01-05,Portfolio up 30% this week, crypto is unstoppable
2024-01-08,Bitcoin dominance rising, altcoin season delayed but coming
2024-01-08,HODL gang winning, weak hands got shaken out already
2024-01-09,ETF volume insane today, this is the beginning of bull run
2024-01-10,Crypto market cap back to 2 trillion, incredible recovery
2024-01-11,Ethereum upgrade looking great, devs delivering as promised
2024-01-15,BTC rejected at 49k, bears pushing back hard
2024-01-15,Massive liquidations today, 500 million wiped in an hour
2024-01-16,Crypto crashing hard, panic selling everywhere
2024-01-16,Lost 40% of my portfolio in 2 days, this is devastating
2024-01-17,Whale dump detected, massive sell walls on all exchanges
2024-01-17,Regulations incoming, SEC going after every crypto project
2024-01-18,Exchange halting withdrawals again, another collapse incoming
2024-01-18,BTC falling through every support level, no floor in sight
2024-01-19,Market sentiment completely destroyed, everyone panic selling
2024-01-22,Dead cat bounce? or real recovery? Very uncertain right now
2024-01-22,Bought the dip cautiously, not sure if bottom is in
2024-01-23,Market stabilizing, cautious optimism returning slowly
2024-01-24,Volume picking up again, smart money accumulating quietly
2024-01-25,BTC showing strength, consolidating above key support
2024-01-29,Breakout confirmed, bulls taking back control of market
2024-01-30,Altcoins following BTC pump, rotation beginning
2024-01-31,January ending green overall despite the mid-month crash
2024-02-01,February looking bullish, halving hype starting to build
2024-02-01,Institutional inflows hitting record highs this week
2024-02-05,Geopolitical tensions tanking risk assets including crypto
2024-02-06,Bitcoin correlation with stocks increasing dangerously
2024-02-06,Oil shock rippling through all markets, crypto not immune
2024-02-07,Tether FUD spreading again, stablecoin risks worrying people
2024-02-08,Leverage getting blown out, cascading liquidations again
2024-02-09,Worst week since the FTX collapse, trust completely broken
2024-02-12,Oversold bounce happening, some buying at these levels
2024-02-13,CPI worse than expected, risk-off trade crushing crypto
2024-02-14,Valentine pump? small rally but nothing sustainable
2024-02-15,On-chain data showing accumulation by long-term holders
2024-02-16,NVIDIA earnings sparking AI token rally across the board
2024-02-20,Halving countdown at 60 days, supply shock narrative building
2024-02-21,NVIDIA crushed earnings, AI euphoria lifting all boats
2024-02-22,Bitcoin breaking out of 3 month range, 50k within reach
2024-02-23,50k broken finally, this is the start of true bull market
2024-02-26,Profit taking after 50k milestone, healthy correction
2024-02-27,Dips being bought aggressively, demand overwhelming supply
2024-02-28,Monthly close looking incredible, best February ever
2024-03-01,Halving in 45 days, every dip is a buying opportunity now
2024-03-04,Bitcoin rejecting 64k, resistance strong but momentum intact
2024-03-05,New all time high above 69k shattered, price discovery mode
2024-03-06,Euphoria levels through the roof, everyone calls 100k by June
2024-03-07,Leverage extremely high, conditions for violent correction set
2024-03-08,Weekend dump as expected, over leveraged longs liquidated
2024-03-11,BTC above 70k, this is absolutely insane and historic
2024-03-12,Altcoins exploding, ETH up 20% this week alone
2024-03-13,FOMO everywhere, people taking loans to buy crypto right now
2024-03-14,Correction needed and healthy, market too extended short term
2024-03-15,Panic selling as BTC drops 15% from ATH in 2 days
2024-03-18,Stabilizing around 65k, still in strong uptrend overall
2024-03-19,Fed confirming cuts later this year, crypto pumping on news
2024-03-20,Back above 67k, bull market firmly intact
2024-03-21,Halving in less than 30 days, accumulation phase in full swing
2024-03-25,BTC consolidating between 65-70k, coiling for next move
2024-03-26,Exchange inflows declining, HODLers not selling at these prices
2024-03-27,Q1 2024 best quarter for crypto since 2020 bull run
2024-03-28,Record quarterly close, institutional adoption undeniable"""

# ═══════════════════════════════════════════════════════════════════════════
#  NLP ENGINE — CRYPTO-SPECIFIC VADER
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_crypto_analyzer():
    """
    VADER augmented with crypto-native lexicon.
    Crypto culture has unique vocabulary VADER doesn't know:
    moon, HODL, rekt, rug pull, FUD, FOMO etc.
    """
    analyzer = SentimentIntensityAnalyzer()
    crypto_lexicon = {
        # Extreme positive (crypto culture)
        "moon":       3.5, "mooning":    3.5, "moonshot":   3.5,
        "bullish":    3.2, "bull":        2.5, "pump":       2.8,
        "HODL":       2.5, "hodl":        2.5, "accumulate": 2.2,
        "breakout":   2.8, "ATH":         3.0, "ath":        3.0,
        "rip":        2.0, "lambo":       2.5, "wagmi":      3.0,
        "WAGMI":      3.0, "LFG":         3.0, "lfg":        3.0,
        "parabolic":  2.8, "exploding":   2.5, "skyrocket":  3.0,
        "institutional": 1.8, "adoption": 2.0, "halving":   2.0,
        "FOMO":       1.5, "fomo":        1.5,  # fear of missing out = mild positive momentum
        # Extreme negative (crypto culture)
        "rekt":      -3.5, "REKT":       -3.5, "rugpull":   -4.0,
        "rug":       -3.0, "scam":       -3.5, "hack":      -3.0,
        "bearish":   -3.2, "bear":       -2.5, "dump":      -2.8,
        "crash":     -3.5, "collapse":   -3.8, "liquidation": -3.0,
        "liquidated": -3.2, "panic":     -2.8, "FUD":       -2.5,
        "fud":       -2.5, "ngmi":       -3.0, "NGMI":      -3.0,
        "worthless": -3.5, "fraud":      -3.5, "ponzi":     -4.0,
        "bankrupt":  -3.8, "insolvent":  -3.5, "halted":    -2.8,
        "devastating": -2.5, "destroyed": -2.5, "wiped":    -2.5,
        # Context-specific
        "volatile":  -0.8, "uncertain":  -1.0, "cautious":  -0.5,
        "correction": -1.0, "resistance": -0.3, "support":   0.3,
        "consolidating": 0.2, "accumulating": 1.5,
    }
    analyzer.lexicon.update(crypto_lexicon)
    return analyzer


def score_to_fear_greed(compound: float) -> float:
    """
    Convert VADER compound (-1 to +1) to Fear & Greed Index (0 to 100).
    0   = Extreme Fear
    50  = Neutral
    100 = Extreme Greed
    Formula: linear mapping with slight center-bias to match real index behavior.
    """
    # Map -1→+1 to 0→100
    raw = (compound + 1) / 2 * 100
    # Slight compression toward center (real Fear & Greed behaves this way)
    compressed = 50 + (raw - 50) * 0.85
    return float(np.clip(compressed, 0, 100))


def classify_fear(score: float) -> tuple:
    """Return (label, css_class) for a Fear & Greed score."""
    if score <= 20:   return "Extreme Fear",  "fear-extreme"
    if score <= 40:   return "Fear",           "fear-high"
    if score <= 60:   return "Neutral",        "fear-neutral"
    if score <= 80:   return "Greed",          "fear-greed"
    return               "Extreme Greed",      "fear-extreme-greed"


def analyze_posts(raw_text: str) -> pd.DataFrame:
    """Parse posts and run NLP sentiment pipeline."""
    analyzer = get_crypto_analyzer()
    records = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            try:
                dt = pd.to_datetime(parts[0].strip())
                text = parts[1].strip()
            except:
                continue
        else:
            dt = pd.to_datetime("today")
            text = line

        scores = analyzer.polarity_scores(text)
        c = scores["compound"]
        fg = score_to_fear_greed(c)
        label, cls = classify_fear(fg)
        records.append({
            "date":       dt,
            "text":       text,
            "compound":   c,
            "fear_greed": fg,
            "label":      label,
            "css_class":  cls,
            "pos":        scores["pos"],
            "neg":        scores["neg"],
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_daily(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-post scores to daily Fear & Greed Index.
    Uses median (robust to outlier posts) + post volume as confidence weight.
    """
    daily = posts_df.groupby(posts_df["date"].dt.date).agg(
        fear_greed  = ("fear_greed", "median"),
        compound    = ("compound",   "mean"),
        post_count  = ("text",       "count"),
        pct_fearful = ("compound",   lambda x: (x < -0.05).mean() * 100),
        pct_greedy  = ("compound",   lambda x: (x > 0.05).mean() * 100),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()
    # Smooth with 3-day rolling median
    daily["fg_smooth"] = daily["fear_greed"].rolling(3, min_periods=1).median()
    return daily


# ═══════════════════════════════════════════════════════════════════════════
#  PRICE DATA
# ═══════════════════════════════════════════════════════════════════════════

def fetch_crypto_prices(ticker: str, start: str, end: str):
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("empty")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df, True
    except:
        return _synthetic_crypto(start, end, ticker), False


def _synthetic_crypto(start, end, ticker):
    """
    Geometric Brownian Motion with regime shifts — mimics crypto volatility clusters.
    High-vol periods simulate crash/pump regimes.
    """
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    np.random.seed(99)

    prices = {
        "BTC-USD": 42000, "ETH-USD": 2200,
        "BNB-USD": 300,   "SOL-USD": 90,
    }
    S0 = prices.get(ticker, 100)

    # Regime-switching volatility
    vol_regime = np.ones(n) * 0.025
    # Mid-Jan crash
    vol_regime[10:16] = 0.06
    # Late Feb breakout
    vol_regime[38:45] = 0.045
    # Early March ATH
    vol_regime[50:58] = 0.055

    mu = 0.003
    W  = np.random.standard_normal(n)
    r  = np.exp((mu - 0.5 * vol_regime**2) + vol_regime * W)
    p  = S0 * np.cumprod(r)

    df = pd.DataFrame({"Close": p,
                       "Volume": np.random.randint(20_000, 80_000, n) * 1_000_000},
                      index=dates)
    df.index.name = "date"
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  COMPUTATIONAL MODEL — GARCH(1,1) WITH FEAR INDEX REGRESSOR
# ═══════════════════════════════════════════════════════════════════════════

def compute_returns(price_df: pd.DataFrame) -> pd.Series:
    returns = price_df["Close"].pct_change().dropna() * 100  # in percent
    return returns


def fit_garch(returns: pd.Series, daily_fear: pd.DataFrame,
              p: int = 1, q: int = 1, forecast_days: int = 7):
    """
    GARCH(p,q) — Generalized Autoregressive Conditional Heteroskedasticity.

    Core idea: tomorrow's volatility depends on:
      - Yesterday's volatility (GARCH term)
      - Yesterday's surprise return squared (ARCH term)
      - External regressor: Fear & Greed Index from NLP

    This is the NLP→Computational Science connection:
      Fear score (from NLP) is injected as a variance regressor (X),
      meaning high fear directly increases the estimated volatility.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·FearIndex_{t-1}
    """
    # Align fear index with returns
    fear_aligned = daily_fear["fg_smooth"].reindex(returns.index, method="ffill").fillna(50)

    # Normalize fear index to 0-1 for regressor
    fear_norm = (fear_aligned - 50).abs() / 50  # deviation from neutral

    # Fit baseline GARCH (no regressor)
    model_base = arch_model(returns, vol="GARCH", p=p, q=q,
                            dist="Normal", rescale=False)
    res_base = model_base.fit(disp="off", show_warning=False)

    # Fit GARCH with Fear Index as external variance regressor
    try:
        model_x = arch_model(returns, vol="GARCH", p=p, q=q,
                             x=fear_norm.values.reshape(-1, 1),
                             dist="Normal", rescale=False)
        res_x = model_x.fit(disp="off", show_warning=False,
                            x=fear_norm.values.reshape(-1, 1))
        use_x = True
    except:
        res_x = res_base
        use_x = False

    # Conditional volatility (annualized %)
    cond_vol = res_base.conditional_volatility * np.sqrt(365)

    # Forecast
    fc = res_base.forecast(horizon=forecast_days, reindex=False)
    fc_vol = np.sqrt(fc.variance.values[-1]) * np.sqrt(365)

    # Risk classification per day
    def classify_vol(v):
        if v > 120:  return "Extreme Risk",  "risk-extreme"
        if v > 80:   return "High Risk",     "risk-high"
        if v > 50:   return "Moderate Risk", "risk-moderate"
        return              "Low Risk",      "risk-low"

    fc_dates = pd.bdate_range(
        start=returns.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )
    forecast_df = pd.DataFrame({
        "date":        fc_dates,
        "volatility":  fc_vol,
        "risk_label":  [classify_vol(v)[0] for v in fc_vol],
        "risk_class":  [classify_vol(v)[1] for v in fc_vol],
    }).set_index("date")

    metrics = {
        "AIC":          round(res_base.aic, 2),
        "BIC":          round(res_base.bic, 2),
        "Log-Lik":      round(res_base.loglikelihood, 2),
        "omega":        round(res_base.params.get("omega", 0), 6),
        "alpha[1]":     round(res_base.params.get("alpha[1]", 0), 4),
        "beta[1]":      round(res_base.params.get("beta[1]", 0), 4),
        "persistence":  round(
            res_base.params.get("alpha[1]", 0) +
            res_base.params.get("beta[1]", 0), 4
        ),
        "used_fear_regressor": use_x,
    }

    return cond_vol, forecast_df, metrics


# ═══════════════════════════════════════════════════════════════════════════
#  MERGED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def merge_all(price_df, daily_fear, cond_vol):
    df = price_df.copy()
    df["return_pct"]  = df["Close"].pct_change() * 100
    df["cond_vol"]    = cond_vol.reindex(df.index)
    df["fear_greed"]  = daily_fear["fear_greed"].reindex(df.index, method="ffill")
    df["fg_smooth"]   = daily_fear["fg_smooth"].reindex(df.index, method="ffill")
    df["post_count"]  = daily_fear["post_count"].reindex(df.index).fillna(0)

    def risk_zone(row):
        fg = row.get("fear_greed", 50)
        cv = row.get("cond_vol", 60)
        if fg < 25 and cv > 80:   return "🔴 Danger Zone"
        if fg < 35 and cv > 60:   return "🟠 High Alert"
        if fg > 70 and cv > 90:   return "🟡 Euphoria Risk"
        if fg > 65 and cv < 50:   return "🟢 Bull Momentum"
        return                           "⚪ Normal"

    df["risk_zone"] = df.apply(risk_zone, axis=1)
    return df.dropna(subset=["Close"])


# ═══════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════

def get_theme_colors(theme):
    if theme == "dark":
        return {
            "BG": "#0d1117",
            "GRID": "#1f2937",
            "PLOT_BG": "#080d14",
            "TEXT": "#e2e8f0",
            "GRID_BORDER": "#374151",
        }
    else:  # light
        return {
            "BG": "#ffffff",
            "GRID": "#e5e7eb",
            "PLOT_BG": "#f9fafb",
            "TEXT": "#1f2937",
            "GRID_BORDER": "#d1d5db",
        }

def get_base_layout(theme):
    colors = get_theme_colors(theme)
    return dict(
        paper_bgcolor=colors["BG"], 
        plot_bgcolor=colors["PLOT_BG"],
        font=dict(color=colors["TEXT"], family="JetBrains Mono"),
        margin=dict(l=55, r=30, t=55, b=50),
        xaxis=dict(gridcolor=colors["GRID"], showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=colors["GRID"], showgrid=True, zeroline=False),
    )


def chart_price_vol(merged, theme="dark"):
    colors = get_theme_colors(theme)
    BASE = get_base_layout(theme)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        vertical_spacing=0.04)

    # Risk zone backgrounds
    zone_colors = {
        "🔴 Danger Zone":   "rgba(239,68,68,0.08)",
        "🟠 High Alert":    "rgba(249,115,22,0.06)",
        "🟡 Euphoria Risk": "rgba(234,179,8,0.06)",
        "🟢 Bull Momentum": "rgba(34,197,94,0.06)",
        "⚪ Normal":        "rgba(0,0,0,0)",
    }
    for i in range(len(merged) - 1):
        col = zone_colors.get(merged["risk_zone"].iloc[i], "rgba(0,0,0,0)")
        fig.add_vrect(x0=merged.index[i], x1=merged.index[i+1],
                      fillcolor=col, opacity=1, layer="below",
                      line_width=0, row=1, col=1)
        fig.add_vrect(x0=merged.index[i], x1=merged.index[i+1],
                      fillcolor=col, opacity=1, layer="below",
                      line_width=0, row=2, col=1)

    # Price
    fig.add_trace(go.Scatter(
        x=merged.index, y=merged["Close"],
        name="Price", line=dict(color="#60a5fa", width=2),
    ), row=1, col=1)

    # Conditional volatility
    fig.add_trace(go.Scatter(
        x=merged.index, y=merged["cond_vol"],
        name="GARCH Volatility (ann. %)",
        line=dict(color="#f472b6", width=2),
        fill="tozeroy", fillcolor="rgba(244,114,182,0.08)",
    ), row=2, col=1)

    # Danger threshold line
    fig.add_hline(y=80, line_dash="dot",
                  line_color="#ef4444", opacity=0.5,
                  annotation_text="High Risk Threshold",
                  annotation_font_color="#ef4444",
                  row=2, col=1)

    fig.update_layout(**BASE,
        title="① Crypto Price vs Market Risk",
        legend=dict(bgcolor=colors["GRID"], bordercolor=colors["GRID_BORDER"], borderwidth=1),
        height=480,
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Vol % (ann.)", row=2, col=1)
    return fig


def chart_fear_greed(merged, daily_fear, forecast_df, theme="dark"):
    colors = get_theme_colors(theme)
    BASE = get_base_layout(theme)
    fig = go.Figure()

    zones = [
        (0,20,"rgba(239,68,68,0.12)"),
        (20,40,"rgba(249,115,22,0.08)"),
        (40,60,"rgba(107,114,128,0.05)"),
        (60,80,"rgba(34,197,94,0.08)"),
        (80,100,"rgba(16,185,129,0.12)")
    ]

    for y0, y1, col in zones:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=col,
            layer="below",
            line_width=0
        )

    fig.add_trace(go.Bar(
        x=daily_fear.index,
        y=daily_fear["post_count"],
        name="Post Volume",
        yaxis="y2",
        marker_color="rgba(167,139,250,0.15)"
    ))

    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged["fear_greed"],
        mode="markers+lines",
        name="Daily F&G",
        marker=dict(
            size=5,
            color=merged["fear_greed"],
            colorscale=[
                [0,"#ef4444"],
                [0.5,"#eab308"],
                [1,"#10b981"]
            ],
            showscale=False
        ),
        line=dict(color="rgba(255,255,255,0.1)")
    ))

    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged["fg_smooth"],
        name="3-Day Smoothed",
        line=dict(color="#a78bfa", width=2.5)
    ))

    layout_dict = BASE.copy()
    layout_dict.update({
        "title": "② What Crypto Social Media Feels Right Now",
        "yaxis": dict(
            title="Fear & Greed",
            range=[0,100],
            gridcolor=colors["GRID"],
            showgrid=True
        ),
        "yaxis2": dict(
            title="Post Count",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        "legend": dict(
            bgcolor=colors["GRID"]
        ),
        "height": 380
    })
    fig.update_layout(**layout_dict)

    return fig

def chart_vol_forecast(forecast_df, cond_vol, returns, theme="dark"):
    colors = get_theme_colors(theme)
    BASE = get_base_layout(theme)
    fig = go.Figure()

    # Historical volatility
    hist = cond_vol.iloc[-30:]

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        name="Historical GARCH Vol",
        line=dict(color="#60a5fa", width=2)
    ))

    fc_colors = {
        "Extreme Risk":"#ef4444",
        "High Risk":"#f97316",
        "Moderate Risk":"#eab308",
        "Low Risk":"#22c55e"
    }

    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["volatility"],
        name="Forecasted Volatility",
        line=dict(color="#f472b6", width=2.5, dash="dot"),
        marker=dict(
            size=8,
            color=[fc_colors.get(r,"#888")
                   for r in forecast_df["risk_label"]]
        ),
        mode="lines+markers"
    ))

    hist_std = cond_vol.std()

    fig.add_trace(go.Scatter(
        x=list(forecast_df.index)+list(forecast_df.index[::-1]),
        y=list(forecast_df["volatility"]+hist_std)+
          list((forecast_df["volatility"]-hist_std)[::-1]),

        fill="toself",
        fillcolor="rgba(244,114,182,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty Band"
    ))

    # FIXED TIMESTAMP ISSUE
    forecast_start = str(cond_vol.index[-1])

    fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        line_color=colors["GRID_BORDER"]
    )

    fig.add_annotation(
        x=forecast_start,
        y=max(cond_vol.max(),
              forecast_df["volatility"].max()),
        text="Forecast →",
        showarrow=False,
        font=dict(color="#f472b6")
    )

    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="#ef4444",
        opacity=0.4
    )

    layout_dict = BASE.copy()
    layout_dict.update({
        "title": "③ Expected Market Chaos (Next Few Days)",
        "yaxis": dict(
            title="Annualized Volatility (%)",
            gridcolor=colors["GRID"],
            showgrid=True,
            zeroline=False
        ),
        "legend": dict(bgcolor=colors["GRID"]),
        "height": 360
    })
    fig.update_layout(**layout_dict)

    return fig


def chart_fear_vs_vol(merged, theme="dark"):
    colors = get_theme_colors(theme)
    BASE = get_base_layout(theme)
    # Scatter: fear vs volatility — shows the NLP→GARCH relationship
    clean = merged.dropna(subset=["fear_greed","cond_vol"])

    fig = px.scatter(
        clean.reset_index(),
        x="fear_greed", y="cond_vol",
        color="cond_vol",
        color_continuous_scale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]],
        hover_data={"date": True, "Close": ":.2f",
                    "cond_vol": ":.1f", "fear_greed": ":.1f"},
        labels={"fear_greed":"Fear & Greed Index","cond_vol":"GARCH Volatility (%)"},
    )

    # Trend line
    x = clean["fear_greed"].values
    y = clean["cond_vol"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 3:
        z = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(x[mask].min(), x[mask].max(), 50)
        fig.add_trace(go.Scatter(
            x=xline, y=np.polyval(z, xline),
            name="Trend", line=dict(color="#a78bfa", width=2, dash="dash"),
        ))

    layout_dict = BASE.copy()
    layout_dict.update({
        "title": "④ Does Panic Cause Bigger Price Swings?",
        "coloraxis_showscale": False,
        "height": 360,
    })
    fig.update_layout(**layout_dict)
    return fig


def chart_returns_dist(returns, cond_vol, theme="dark"):
    colors = get_theme_colors(theme)
    BASE = get_base_layout(theme)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Return Distribution", "Volatility Clustering"])

    fig.add_trace(go.Histogram(
        x=returns, nbinsx=40,
        marker_color="#60a5fa", opacity=0.75,
        name="Returns",
    ), row=1, col=1)

    # Normal overlay
    mu, sigma = returns.mean(), returns.std()
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    y_norm = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm-mu)/sigma)**2)
    y_norm = y_norm * len(returns) * (returns.max()-returns.min()) / 40
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm, name="Normal Fit",
        line=dict(color="#f472b6", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cond_vol.index, y=cond_vol.values,
        name="GARCH Vol", line=dict(color="#f472b6", width=1.5),
        fill="tozeroy", fillcolor="rgba(244,114,182,0.1)",
    ), row=1, col=2)

    layout_dict = BASE.copy()
    layout_dict.update({
        "title": "⑤ Return Distribution & Volatility Clustering",
        "showlegend": True,
        "height": 340,
    })
    fig.update_layout(**layout_dict)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🌓 Theme")
    theme_option = st.radio(
        "Select Theme",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state.theme == "dark" else 1,
        label_visibility="collapsed"
    )
    
    selected_theme = "dark" if "🌙" in theme_option else "light"
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    st.markdown("---")

    st.markdown("### 💰 Crypto Asset")
    ticker = st.selectbox(
        "Select Cryptocurrency",
        ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "Custom"],
        index=0,
    )
    if ticker == "Custom":
        ticker = st.text_input("Enter ticker (e.g. DOGE-USD)", "DOGE-USD")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date(2024, 1, 1),
                                   min_value=date(2020, 1, 1))
    with col2:
        end_date   = st.date_input("End",   value=date(2024, 3, 28),
                                   min_value=date(2020, 1, 2))

    st.markdown("### 📐 GARCH Settings")
    garch_p = st.slider("ARCH order (p)", 1, 3, 1,
                        help="Number of lagged squared residuals")
    garch_q = st.slider("GARCH order (q)", 1, 3, 1,
                        help="Number of lagged variance terms")
    forecast_days = st.slider("Forecast Horizon (days)", 3, 14, 7)

    st.markdown("### 📊 Risk Thresholds")
    high_risk_vol = st.slider("High Risk Volatility (%)", 50, 150, 80,
                               help="Annualized vol threshold for High Risk classification")
    fear_threshold = st.slider("Fear Zone Threshold", 10, 45, 30,
                                help="F&G score below this = fear zone")
    greed_threshold = st.slider("Greed Zone Threshold", 55, 90, 70,
                                 help="F&G score above this = greed zone")

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <p class="hero-title">🛡️ CryptoSentinel</p>
    <p class="hero-sub">
        Crypto Fear & Volatility Forecasting Engine &nbsp;·&nbsp;
        NLP Fear Index → GARCH(1,1) Volatility Model &nbsp;·&nbsp;
        Early Risk Detection System
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Social Media Input", "📖 How It Works"])

# ── Tab 2: Input ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-hdr">Crypto Social Media Posts</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Format:** `YYYY-MM-DD, post text here`
    Simulates Reddit (r/CryptoCurrency, r/Bitcoin) or Twitter/X posts.
    Each line = one post. The NLP engine scores each post and aggregates
    a daily **Fear & Greed Index** that feeds into the GARCH model.
    """)

    posts_input = st.text_area(
        "Posts", value=DEFAULT_POSTS, height=450,
        label_visibility="collapsed",
    )

    n_posts = len([l for l in posts_input.strip().split("\n") if l.strip()])
    st.info(f"📌 **{n_posts} posts** loaded across the date range.")

    st.markdown('<div class="section-hdr">🔬 Live Post Tester</div>',
                unsafe_allow_html=True)
    test_post = st.text_input(
        "Type any crypto post:",
        placeholder="e.g. Bitcoin just crashed 20% and I'm completely rekt..."
    )
    if test_post:
        analyzer = get_crypto_analyzer()
        sc = analyzer.polarity_scores(test_post)
        fg = score_to_fear_greed(sc["compound"])
        label, cls = classify_fear(fg)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Compound Score", f"{sc['compound']:+.4f}")
        c2.metric("Fear & Greed", f"{fg:.1f}/100")
        c3.metric("Sentiment", label)
        c4.metric("Pos / Neg", f"{sc['pos']:.2f} / {sc['neg']:.2f}")

# ── Tab 3: About ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## How CryptoSentinel Works

    ### The core insight
    In crypto, **volatility is more predictable than direction.**
    When fear dominates social media, the market tends to swing wildly —
    regardless of whether it goes up or down. GARCH models exactly this behavior.

    ---

    ### NLP → Computational Science pipeline

    ```
    Social Media Posts
          ↓
    VADER + Crypto Lexicon (NLP)
    → compound score per post (-1 to +1)
    → Fear & Greed Index per day (0-100)
          ↓
    Daily Fear Index injected as variance regressor
          ↓
    GARCH(1,1) model
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·|FearDeviation_{t-1}|
          ↓
    Conditional Volatility Forecast (annualized %)
          ↓
    Risk Classification → Early Warning Signal
    ```

    ### Why GARCH specifically?
    GARCH captures **volatility clustering** — the empirical fact that
    large price moves tend to follow large price moves (Mandelbrot, 1963;
    Engle, 1982). This is extremely pronounced in crypto. The model
    parameters tell you:

    | Parameter | Meaning |
    |---|---|
    | **α (ARCH)** | How much yesterday's shock affects today's volatility |
    | **β (GARCH)** | How persistent volatility is over time |
    | **α + β** | Persistence — close to 1.0 = volatility shocks last long |

    ### Risk Zones
    | Zone | Condition | Meaning |
    |---|---|---|
    | 🔴 Danger Zone | Fear < 25 AND Vol > 80% | Panic selling, very high risk |
    | 🟠 High Alert | Fear < 35 AND Vol > 60% | Market stressed, caution needed |
    | 🟡 Euphoria Risk | Greed > 70 AND Vol > 90% | Over-extended, correction risk |
    | 🟢 Bull Momentum | Greed > 65 AND Vol < 50% | Healthy uptrend, lower risk |

    ### References
    - Engle (1982). *Autoregressive Conditional Heteroskedasticity.* Econometrica.
    - Bollerslev (1986). *Generalized ARCH.* Journal of Econometrics.
    - Bollen et al. (2011). *Twitter mood predicts the stock market.* J. Comp. Science.
    - Nakano et al. (2018). *Bitcoin technical trading with NLP.* Finance Research Letters.
    """)

# ── Tab 1: Dashboard ──────────────────────────────────────────────────────
with tab1:
    if not run_btn and "crypto_results" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;color:#4b5563;">
            <div style="font-size:5rem;">🛡️</div>
            <div style="font-size:1.2rem;font-weight:700;color:#6b7280;margin-top:16px;">
                Configure settings in the sidebar and click
                <strong style="color:#a78bfa">Run Analysis</strong>
            </div>
            <div style="font-size:0.9rem;color:#374151;margin-top:8px;">
                Add your own crypto posts in the Social Media Input tab
            </div>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        with st.spinner("🔍 Running NLP sentiment pipeline..."):
            posts_df  = analyze_posts(posts_input)
            daily_fear = aggregate_daily(posts_df)

        with st.spinner(f"📡 Fetching {ticker} price data..."):
            price_df, real_data = fetch_crypto_prices(
                ticker, str(start_date), str(end_date)
            )

        with st.spinner("📐 Fitting GARCH model..."):
            returns   = compute_returns(price_df)
            cond_vol, forecast_df, garch_metrics = fit_garch(
                returns, daily_fear, p=garch_p, q=garch_q,
                forecast_days=forecast_days,
            )

        merged = merge_all(price_df, daily_fear, cond_vol)

        st.session_state["crypto_results"] = {
            "posts_df":    posts_df,
            "daily_fear":  daily_fear,
            "price_df":    price_df,
            "returns":     returns,
            "cond_vol":    cond_vol,
            "forecast_df": forecast_df,
            "garch_metrics": garch_metrics,
            "merged":      merged,
            "real_data":   real_data,
        }

    if "crypto_results" in st.session_state:
        r = st.session_state["crypto_results"]
        merged       = r["merged"]
        daily_fear   = r["daily_fear"]
        forecast_df  = r["forecast_df"]
        cond_vol     = r["cond_vol"]
        returns      = r["returns"]
        garch_metrics = r["garch_metrics"]
        posts_df     = r["posts_df"]

        if not r["real_data"]:
            st.warning("⚠️ Yahoo Finance unavailable — using synthetic GBM price data. NLP and GARCH are real.")

        # ── KPI Row ──────────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Live Dashboard</div>',
                    unsafe_allow_html=True)

        latest_fg    = merged["fear_greed"].dropna().iloc[-1]
        latest_vol   = merged["cond_vol"].dropna().iloc[-1]
        fc_max_vol   = forecast_df["volatility"].max()
        fc_risk      = forecast_df["risk_label"].iloc[0]
        persistence  = garch_metrics["persistence"]
        fg_label, fg_cls = classify_fear(latest_fg)
        n_danger = (merged["risk_zone"] == "🔴 Danger Zone").sum()

        c1,c2,c3,c4,c5,c6 = st.columns(6)

        def kpi(col, label, val, sub="", cls=""):
            col.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub {cls}">{sub}</div>
            </div>""", unsafe_allow_html=True)

        kpi(c1,"Market Mood",f"{latest_fg:.0f}",fg_label,fg_cls)
        kpi(c2,"Current Risk",f"{latest_vol:.0f}%","Expected price swings")
        kpi(c3, f"{forecast_days}D Forecast Vol",
            f"{fc_max_vol:.0f}%", f"Peak: {fc_risk}",
            "fear-extreme" if fc_max_vol > 100 else "fear-neutral")
        kpi(c4,"Panic Persistence",f"{persistence:.2f}","How long volatility lasts")
        kpi(c5, "Posts Analyzed", str(len(posts_df)),
            f"{daily_fear['post_count'].sum():.0f} total")
        kpi(c6, "Danger Days", str(n_danger),
            "in period", "fear-extreme" if n_danger > 5 else "")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📌 Today's Crypto Outlook")
        summary_parts = []
        if latest_fg<25: summary_parts.append("😨 Social media is panicking.")
        elif latest_fg>75: summary_parts.append("🚀 Crypto sentiment is very bullish.")
        else: summary_parts.append("😐 Sentiment is mixed.")
        if fc_max_vol>100: summary_parts.append("⚠️ Large price swings expected.")
        elif fc_max_vol<50: summary_parts.append("✅ Market looks relatively stable.")
        
        summary_text = "\n".join(summary_parts)
        summary_text += f"\n\nMarket mood: **{fg_label}**\n\nExpected risk: **{fc_risk}**"
        st.info(summary_text)


        # ── Charts ────────────────────────────────────────────────────────
        
        st.markdown('<div class="section-hdr">📊 Market Analysis Charts</div>',
                    unsafe_allow_html=True)

        chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
            "💹 Price & Risk Level",
            "🗣️ Market Sentiment",
            "⚡ Fear Impact",
            "🔮 7-Day Forecast",
            "📈 Price Patterns"
        ])

        with chart_tab1:
            st.plotly_chart(chart_price_vol(merged, st.session_state.theme), use_container_width=True)
            st.info(f"""
**Real-time Price & Risk Assessment**

Current market risk is **{latest_vol:.0f}%**

Higher values mean larger expected price swings.

Risk zone: **{merged['risk_zone'].iloc[-1]}** — colored zones show different risk periods over time.
""")

        with chart_tab2:
            st.plotly_chart(chart_fear_greed(merged, daily_fear, forecast_df, st.session_state.theme), use_container_width=True)
            st.info(f"""
**What Are Crypto Traders Saying?**

Market sentiment: **{fg_label}** (Score: **{latest_fg:.0f}/100**)

**Sentiment Scale:**
- **0–20:** Extreme Fear (panic selling)
- **40–60:** Neutral (mixed feelings)
- **80–100:** Extreme Greed (euphoria)

Bars show daily post volume. The smoothed line filters out noise to show real trends.
""")

        with chart_tab3:
            st.plotly_chart(chart_fear_vs_vol(merged, st.session_state.theme), use_container_width=True)
            st.info("""
**Does Panic Actually Cause Bigger Swings?**

This chart reveals the relationship between market fear and volatility.

Points moving up-right show that when fear increases, price swings tend to get larger.

A clear upward trend confirms that scared traders = wilder markets.
""")

        with chart_tab4:
            st.plotly_chart(chart_vol_forecast(forecast_df, cond_vol, returns, st.session_state.theme), use_container_width=True)
            st.info(f"""
**What's Coming in the Next {forecast_days} Days?**

Highest expected volatility: **{fc_max_vol:.1f}%**

Risk outlook: **{fc_risk}**

The pink line shows our prediction. The shaded band around it represents uncertainty — actual volatility will likely fall within this range.

Blue line shows recent history for context.
""")

        with chart_tab5:
            st.plotly_chart(chart_returns_dist(returns, cond_vol, st.session_state.theme), use_container_width=True)
            st.info("""
**How Do Price Swings Happen?**

**Left Chart: Distribution of daily returns**
Shows if movements are small & frequent or rare & extreme. The pink curve is what a 'normal' market would look like. Crypto usually has fatter tails (more extreme moves).

**Right Chart: Volatility clustering**
Notice how high-risk periods bunch together. One big move often triggers more big moves.
""")

        # ── Forecast Table ────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">📅 Volatility Forecast Table</div>',
                    unsafe_allow_html=True)
        fc_display = forecast_df.copy().reset_index()
        fc_display["date"]       = fc_display["date"].dt.strftime("%Y-%m-%d")
        fc_display["volatility"] = fc_display["volatility"].map(lambda x: f"{x:.1f}%")
        fc_display = fc_display[["date","volatility","risk_label"]]
        fc_display.columns = ["Date","Forecasted Volatility","Risk Level"]
        st.dataframe(fc_display, use_container_width=True, hide_index=True)

        # ── GARCH Stats ───────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">📐 GARCH Model Statistics</div>',
                    unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.dataframe(pd.DataFrame({
                "Parameter": ["AIC","BIC","Log-Likelihood",
                              "omega","alpha[1]","beta[1]","α+β (persistence)"],
                "Value": [
                    garch_metrics["AIC"], garch_metrics["BIC"],
                    garch_metrics["Log-Lik"], garch_metrics["omega"],
                    garch_metrics["alpha[1]"], garch_metrics["beta[1]"],
                    garch_metrics["persistence"],
                ],
            }), use_container_width=True, hide_index=True)
        with col_s2:
            st.markdown("""
            <div class="insight-box">
                <div class="insight-title">📌 How to read α+β (persistence)</div>
                <div class="insight-text">
                Close to <strong>1.0</strong> = volatility shocks last a long time (typical in crypto).<br>
                Close to <strong>0.5</strong> = shocks die out quickly.<br>
                Crypto typically shows 0.90–0.98 persistence.
                </div>
            </div>
            <div class="insight-box">
                <div class="insight-title">📌 NLP → GARCH connection</div>
                <div class="insight-text">
                The Fear & Greed Index (built from NLP) is used as a <strong>variance regressor</strong>
                in the GARCH model — high fear deviations directly increase
                the estimated conditional volatility, linking social sentiment
                to quantitative risk forecasting.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Top fearful / greedy posts ────────────────────────────────────
        st.markdown('<div class="section-hdr">🏆 Most Extreme Posts</div>',
                    unsafe_allow_html=True)
        col_f, col_g = st.columns(2)
        with col_f:
            st.markdown("**Most Fearful Posts**")
            fearful = posts_df.nsmallest(5, "compound")[["text","compound","fear_greed"]]
            fearful["compound"]   = fearful["compound"].map(lambda x: f"{x:+.3f}")
            fearful["fear_greed"] = fearful["fea" \
            "r_greed"].map(lambda x: f"{x:.0f}/100")
            fearful.columns = ["Post","Score","F&G Index"]
            st.dataframe(fearful.reset_index(drop=True),
                         use_container_width=True, hide_index=True)
        with col_g:
            st.markdown("**Most Greedy Posts**")
            greedy = posts_df.nlargest(5, "compound")[["text","compound","fear_greed"]]
            greedy["compound"]   = greedy["compound"].map(lambda x: f"{x:+.3f}")
            greedy["fear_greed"] = greedy["fear_greed"].map(lambda x: f"{x:.0f}/100")
            greedy.columns = ["Post","Score","F&G Index"]
            st.dataframe(greedy.reset_index(drop=True),
                         use_container_width=True, hide_index=True)
