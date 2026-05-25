"""
=============================================================================
 Volasense — Crypto Fear & Volatility Forecasting Engine
=============================================================================
 NLP   : VADER + custom crypto lexicon → Daily Fear/Greed Index (0–100)
 Model : GARCH(1,1) — industry-standard volatility forecasting
 Field : Cryptocurrency / Finance

 How NLP connects to Computational Science:
   Crypto posts/news → Fear Score → injected as external regressor
   into GARCH model → volatility forecast → risk classification

 Run: streamlit run main.py
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
from datetime import date, timedelta, datetime
import requests

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from arch import arch_model
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Volasense",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state initialization
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "reddit_posts" not in st.session_state:
    st.session_state.reddit_posts = None


FEATHER_ICONS = {
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>',
    "settings": (
        '<circle cx="12" cy="12" r="3"></circle>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83'
        ' 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33'
        ' 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2'
        ' 2 2 0 0 1-2-2v-.09a1.65 1.65 0 0 0-1-1.51'
        ' 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0'
        ' 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82'
        ' 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2'
        ' 2 2 0 0 1 2-2h.09a1.65 1.65 0 0 0 1.51-1'
        ' 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83'
        ' 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33'
        'H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2'
        ' 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51'
        ' 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0'
        ' 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82'
        'V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2'
        ' 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>'
    ),
    "sun": (
        '<circle cx="12" cy="12" r="5"></circle>'
        '<line x1="12" y1="1" x2="12" y2="3"></line>'
        '<line x1="12" y1="21" x2="12" y2="23"></line>'
        '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>'
        '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>'
        '<line x1="1" y1="12" x2="3" y2="12"></line>'
        '<line x1="21" y1="12" x2="23" y2="12"></line>'
        '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>'
        '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>'
    ),
    "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>',
    "dollar-sign": (
        '<line x1="12" y1="1" x2="12" y2="23"></line>'
        '<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7H14a3.5 3.5 0 0 1 0 7H6"></path>'
    ),
    "crosshair": (
        '<circle cx="12" cy="12" r="10"></circle>'
        '<line x1="22" y1="12" x2="18" y2="12"></line>'
        '<line x1="6" y1="12" x2="2" y2="12"></line>'
        '<line x1="12" y1="2" x2="12" y2="6"></line>'
        '<line x1="12" y1="18" x2="12" y2="22"></line>'
    ),
    "wifi": (
        '<path d="M5 12.55a11 11 0 0 1 14.08 0"></path>'
        '<path d="M1.42 9a16 16 0 0 1 21.16 0"></path>'
        '<path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path>'
        '<line x1="12" y1="20" x2="12.01" y2="20"></line>'
    ),
    "bar-chart-2": (
        '<line x1="18" y1="20" x2="18" y2="10"></line>'
        '<line x1="12" y1="20" x2="12" y2="4"></line>'
        '<line x1="6" y1="20" x2="6" y2="14"></line>'
    ),
    "edit-3": (
        '<path d="M12 20h9"></path>'
        '<path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z"></path>'
    ),
    "book-open": (
        '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>'
        '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="8"></circle>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"></line>'
    ),
    "activity": (
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>'
    ),
    "table": (
        '<path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h6m-6 0v18m6-18h4a2 2 0 0 1 2 2v4m-6-6v18m6-12H3m18 0v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9m18 0H3"></path>'
    ),
    "alert-triangle": (
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>'
        '<line x1="12" y1="9" x2="12" y2="13"></line>'
        '<line x1="12" y1="17" x2="12.01" y2="17"></line>'
    ),
    "trending-up": (
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>'
        '<polyline points="17 6 23 6 23 12"></polyline>'
    ),
}


def feather_icon(name: str, size: int = 18, extra_class: str = "") -> str:
    body = FEATHER_ICONS.get(name)
    if not body:
        return ""
    cls = ("fi " + extra_class).strip()
    return (
        f'<svg class="{cls}" xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )


def get_theme_css(theme):
    if theme == "dark":
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
        :root { color-scheme: dark; }

        html, body, .stApp, div[data-testid="stAppViewContainer"] {
            background: #0b0f16;
            color: #e2e8f0;
        }
        div[data-testid="stHeader"] {
            background: rgba(11, 15, 22, 0.7);
            backdrop-filter: blur(6px);
        }
        div.block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .hero {
            background: linear-gradient(135deg, #0b1220 0%, #0b2a4a 50%, #063d35 100%);
            border: 1px solid #1e3a8a;
            border-radius: 20px;
            padding: 32px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #14b8a6, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 8px 0;
        }
        .hero-sub { color: #94a3b8; font-size: 0.95rem; margin: 0; }

        .stTabs [data-baseweb="tab-list"] {
            background: #0f172a;
            border: 1px solid #1f2937;
            border-radius: 999px;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #cbd5f5;
            border-radius: 999px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #1f2937;
            color: #f8fafc;
        }

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
            border-left: 3px solid #38bdf8;
            padding-left: 10px; margin: 28px 0 14px 0;
        }
        .insight-box {
            background: #0d1117;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 10px;
        }
        .insight-title { color: #38bdf8; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px; }
        .insight-text  { color: #d1d5db; font-size: 0.88rem; line-height: 1.5; }

        div[data-testid="stSidebar"],
        section[data-testid="stSidebar"] {
            background: #0a0a12;
        }
        div[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #14b8a6);
            color: white; border: none; border-radius: 12px;
            padding: 12px 28px; font-weight: 700;
            font-size: 0.95rem; width: 100%;
        }
        .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

        .fi { display: inline-block; vertical-align: -0.18em; margin-right: 0.45rem; }
        .hero-title .fi { margin-right: 0.55rem; }

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] > div {
            background-color: #0d1117 !important;
            color: #e2e8f0 !important;
            border-color: #1f2937 !important;
        }
        </style>
        """
    else:  # Light mode
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
        :root { color-scheme: light; }

        html, body, .stApp, div[data-testid="stAppViewContainer"] {
            background: #f5f7fb;
            color: #0b0b0b;
        }
        div[data-testid="stHeader"] {
            background: rgba(245, 247, 251, 0.8);
            backdrop-filter: blur(6px);
        }
        div.block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #eef6ff 50%, #eefaf3 100%);
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 32px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #2563eb, #14b8a6, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 8px 0;
        }
        .hero-sub { color: #475569; font-size: 0.98rem; margin: 0; }

        .stTabs [data-baseweb="tab-list"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 999px;
            padding: 6px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }
        .stTabs [data-baseweb="tab"] {
            color: #475569;
            border-radius: 999px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #f1f5f9;
            color: #0b0b0b;
        }
        .stTabs [data-baseweb="tab"] span {
            font-weight: 600;
        }

        .kpi-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e7e5f4;
            transition: border-color 0.2s;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .kpi-label { color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; }
        .kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #0b0b0b; }
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
            font-size: 1.05rem; font-weight: 800; color: #0b0b0b;
            border-left: 4px solid #2563eb;
            padding-left: 12px; margin: 30px 0 16px 0;
        }
        .insight-box {
            background: #ffffff;
            border: 1px solid #e7e5f4;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 10px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        }
        .insight-title { color: #2563eb; font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; }
        .insight-text  { color: #0b0b0b; font-size: 0.92rem; line-height: 1.6; }

        div[data-testid="stSidebar"],
        section[data-testid="stSidebar"] {
            background: #f5f7fb;
            color: #0b0b0b !important;
        }
        /* Sidebar drawer / collapsed-control surfaces (mobile + narrow layouts) */
        div[data-testid="stSidebarNav"],
        div[data-testid="stSidebarHeader"],
        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarCollapsedControl"],
        div[data-testid="stSidebarCollapsedControl"] > div,
        div[data-testid="collapsedControl"],
        div[data-testid="collapsedControl"] > div {
            background: #f5f7fb !important;
        }
        div[data-testid="stSidebarCollapsedControl"] button {
            background: #f5f7fb !important;
            color: #0b0b0b !important;
            border-color: #e5e7eb !important;
        }
        div[data-testid="collapsedControl"] button,
        button[data-testid="collapsedControl"] {
            background: #f5f7fb !important;
            color: #0b0b0b !important;
            border-color: #e5e7eb !important;
        }
        div[data-testid="stSidebarCollapsedControl"] svg {
            fill: #0b0b0b !important;
            color: #0b0b0b !important;
        }
        div[data-testid="collapsedControl"] svg,
        button[data-testid="collapsedControl"] svg {
            fill: #0b0b0b !important;
            color: #0b0b0b !important;
        }
        /* BaseWeb drawer used by Streamlit sidebar overlay */
        div[data-baseweb="drawer"] > div {
            background: #f5f7fb !important;
            color: #0b0b0b !important;
        }
        div[data-baseweb="drawer"] {
            color: #0b0b0b !important;
        }
        div[data-baseweb="drawer"] * {
            color: #0b0b0b !important;
        }
        div[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] * {
            color: #0b0b0b !important;
        }
        div[data-testid="stSidebarNav"] * {
            color: #0b0b0b !important;
        }
        div[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
        div[data-testid="stSidebar"] hr {
            border-color: #e5e7eb;
        }
        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #14b8a6);
            color: white; border: none; border-radius: 12px;
            padding: 12px 28px; font-weight: 700;
            font-size: 0.95rem; width: 100%;
        }
        .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

        .fi { display: inline-block; vertical-align: -0.18em; margin-right: 0.45rem; }
        .hero-title .fi { margin-right: 0.55rem; }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e7e5f4;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetricLabel"] { color: #64748b; }
        div[data-testid="stMetricValue"] { color: #0b0b0b; }
        div[data-testid="stMetricDelta"] { color: #0b0b0b; }

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #0b0b0b !important;
            border-color: #e5e7eb !important;
        }
        [data-baseweb="input"] input::placeholder,
        [data-baseweb="textarea"] textarea::placeholder {
            color: #94a3b8 !important;
        }

        .stMarkdown p, .stMarkdown li, .stMarkdown span,
        label, .stText, .stCaption {
            color: #0b0b0b;
        }

        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] span,
        div[data-testid="stSidebar"] h1,
        div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3,
        div[data-testid="stSidebar"] h4,
        div[data-testid="stSidebar"] h5,
        div[data-testid="stSidebar"] h6,
        div[data-testid="stSidebar"] li,
        div[data-testid="stSidebar"] small {
            color: #0b0b0b !important;
        }

        div[data-testid="stSidebar"] [data-baseweb] {
            color: #0b0b0b !important;
        }

        .stRadio label,
        .stSelectbox label,
        .stMultiSelect label,
        .stDateInput label,
        .stSlider label,
        .stTextInput label,
        .stTextArea label,
        .stToggle label {
            color: #0b0b0b !important;
        }
        </style>
        """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  REDDIT LIVE FETCH (No API Key Required)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_reddit_posts(
    subreddits=["CryptoCurrency", "Bitcoin"],
    limit=100,
    sort="hot",
    max_posts=None,        # None = unlimited (fetch all available pages)
    progress_callback=None, # optional fn(fetched_so_far, sub) for UI updates
    date_from=None,        # date: only keep posts on or after this date
    date_to=None,          # date: only keep posts on or before this date
):
    """
    Fetch posts from crypto subreddits using Reddit's public JSON endpoint.
    No API credentials required — uses Reddit's free public JSON API.

    Pagination: Reddit returns up to 100 posts per request. This function
    follows the `after` cursor across multiple pages to retrieve as many
    posts as are available (up to `max_posts` per subreddit if set).

    Returns a CSV string in the same format as DEFAULT_POSTS: YYYY-MM-DD,text
    """
    headers = {"User-Agent": "Volasense/1.0 (research project)"}
    # Reddit clamps each page to 100 posts max
    page_size = min(100, limit if max_posts is None else 100)
    lines = []
    seen_ids = set()  # deduplicate across pages

    for sub in subreddits:
        after = None          # pagination cursor
        sub_count = 0         # posts collected for this subreddit
        consecutive_empties = 0

        while True:
            # Stop if we've hit the per-subreddit cap
            if max_posts is not None and sub_count >= max_posts:
                break

            params = f"limit={page_size}"
            if after:
                params += f"&after={after}"

            try:
                url = f"https://www.reddit.com/r/{sub}/{sort}.json?{params}"
                res = requests.get(url, headers=headers, timeout=15)
                if res.status_code == 429:
                    # Rate-limited — back off and retry once
                    import time; time.sleep(2)
                    res = requests.get(url, headers=headers, timeout=15)
                if res.status_code != 200:
                    break

                data = res.json()
                children = data["data"]["children"]
                after    = data["data"].get("after")  # next page cursor

                if not children:
                    consecutive_empties += 1
                    if consecutive_empties >= 2:
                        break
                    continue

                consecutive_empties = 0
                page_added = 0

                for post in children:
                    p    = post["data"]
                    pid  = p.get("id", "")

                    # Skip duplicates (can appear across hot/new pages)
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    date_str = datetime.utcfromtimestamp(
                        p["created_utc"]
                    ).strftime("%Y-%m-%d")

                    # Combine title + selftext snippet for richer sentiment
                    title    = p.get("title", "").replace("\n", " ").replace(",", " ").strip()
                    selftext = p.get("selftext", "")[:120].replace("\n", " ").replace(",", " ").strip()
                    text     = f"{title}. {selftext}".strip(". ") if selftext else title

                    if text:
                        post_date = datetime.utcfromtimestamp(p["created_utc"]).date()
                        in_range = (
                            (date_from is None or post_date >= date_from) and
                            (date_to   is None or post_date <= date_to)
                        )
                        if in_range:
                            lines.append(f"{date_str},{text}")
                            sub_count  += 1
                            page_added += 1

                    if max_posts is not None and sub_count >= max_posts:
                        break

                if progress_callback:
                    progress_callback(len(lines), sub)

                # No more pages available
                if not after:
                    break

                # Reddit throttle — be polite
                import time; time.sleep(0.6)

            except Exception:
                break

    return "\n".join(lines) if lines else None


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

    # Reindex fear data — forward fill, then backward fill so no NaNs remain
    df["fear_greed"]  = daily_fear["fear_greed"].reindex(df.index, method="ffill")
    df["fg_smooth"]   = daily_fear["fg_smooth"].reindex(df.index, method="ffill")
    df["post_count"]  = daily_fear["post_count"].reindex(df.index).fillna(0)

    # If fear_greed is all NaN (date mismatch), fill with neutral 50
    if df["fear_greed"].isna().all():
        df["fear_greed"] = 50.0
        df["fg_smooth"]  = 50.0
    else:
        # Backward fill any remaining NaNs at the start
        df["fear_greed"] = df["fear_greed"].bfill().fillna(50.0)
        df["fg_smooth"]  = df["fg_smooth"].bfill().fillna(50.0)

    def risk_zone(row):
        fg = row.get("fear_greed", 50)
        cv = row.get("cond_vol", 60)
        if fg < 25 and cv > 80:   return "Danger Zone"
        if fg < 35 and cv > 60:   return "High Alert"
        if fg > 70 and cv > 90:   return "Euphoria Risk"
        if fg > 65 and cv < 50:   return "Bull Momentum"
        return                           "Normal"

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
        "Danger Zone":   "rgba(239,68,68,0.08)",
        "High Alert":    "rgba(249,115,22,0.06)",
        "Euphoria Risk": "rgba(234,179,8,0.06)",
        "Bull Momentum": "rgba(34,197,94,0.06)",
        "Normal":        "rgba(0,0,0,0)",
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
        line=dict(color="#14b8a6", width=2),
        fill="tozeroy", fillcolor="rgba(20,184,166,0.10)",
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
        marker_color="rgba(37,99,235,0.14)"
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
        line=dict(color="#2563eb", width=2.5)
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
        line=dict(color="#14b8a6", width=2.5, dash="dot"),
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
        fillcolor="rgba(20,184,166,0.10)",
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
        font=dict(color="#14b8a6")
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
            name="Trend", line=dict(color="#2563eb", width=2, dash="dash"),
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
        line=dict(color="#14b8a6", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cond_vol.index, y=cond_vol.values,
        name="GARCH Vol", line=dict(color="#14b8a6", width=1.5),
        fill="tozeroy", fillcolor="rgba(20,184,166,0.10)",
    ), row=1, col=2)

    layout_dict = BASE.copy()
    layout_dict.update({
        "title": "⑤ Return Distribution & Volatility Clustering",
        "showlegend": True,
        "height": 340,
    })
    fig.update_layout(**layout_dict)
    return fig

def chart_prediction_vs_actual(returns, cond_vol, forecast_df, theme="dark"):
    colors = get_theme_colors(theme)
    BASE = get_base_layout(theme)

    fig = go.Figure()

    # Actual realized volatility (7-day rolling volatility)
    actual_vol = returns.rolling(7).std() * np.sqrt(365)

    fig.add_trace(go.Scatter(
        x=actual_vol.index,
        y=actual_vol,
        name="Actual Volatility",
        line=dict(color="#60a5fa", width=2)
    ))

    # Predicted historical volatility from GARCH
    fig.add_trace(go.Scatter(
        x=cond_vol.index,
        y=cond_vol,
        name="Predicted (GARCH)",
        line=dict(color="#14b8a6", width=2)
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["volatility"],
        name="Forecast",
        mode="lines+markers",
        line=dict(color="#f59e0b", dash="dot", width=2),
        marker=dict(size=8)
    ))

    fig.add_vline(
        x=cond_vol.index[-1],
        line_dash="dash",
        line_color="gray"
    )

    fig.add_annotation(
        x=cond_vol.index[-1],
        y=max(
            actual_vol.max(skipna=True),
            cond_vol.max(),
            forecast_df["volatility"].max()
        ),
        text="Forecast Starts",
        showarrow=False,
        font=dict(color="#f59e0b")
    )

    layout = BASE.copy()
    layout.update({
        "title": "⑥ Prediction vs Actual Volatility",
        "height": 400,
        "yaxis": dict(
            title="Annualized Volatility (%)",
            gridcolor=colors["GRID"]
        ),
        "legend": dict(
            bgcolor=colors["GRID"]
        )
    })

    fig.update_layout(**layout)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"## {feather_icon('settings', 18)} Volasense", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"### {feather_icon('sun', 16)} Theme", unsafe_allow_html=True)
    theme_option = st.radio(
        "Select Theme",
        ["Dark Mode", "Light Mode"],
        index=0 if st.session_state.theme == "dark" else 1,
        label_visibility="collapsed"
    )
    
    selected_theme = "dark" if theme_option.startswith("Dark") else "light"
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    st.markdown("---")

    st.markdown(f"### {feather_icon('dollar-sign', 16)} Crypto Asset", unsafe_allow_html=True)
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
                                   min_value=date(2005, 1, 1))
    with col2:
        end_date   = st.date_input("End", value=date.today(),
                                   min_value=date(2005, 1, 2))

    st.markdown(f"### {feather_icon('crosshair', 16)} Choose Analysis Style", unsafe_allow_html=True)

    mode = st.selectbox(
        "Analysis Mode",
        ["Beginner (Recommended)",
         "Balanced",
         "Early Warnings",
         "Advanced"],
        help="Choose how sensitive the crypto warning system should be"
    )

    if mode == "Beginner (Recommended)":
        garch_p, garch_q, forecast_days = 1, 1, 7
        high_risk_vol, fear_threshold, greed_threshold = 80, 30, 70
        st.success("Easy mode: Stable forecasts and simple explanations.")

    elif mode == "Balanced":
        garch_p, garch_q, forecast_days = 1, 1, 10
        high_risk_vol, fear_threshold, greed_threshold = 70, 35, 65
        st.info("Balanced mode: More responsive to market changes.")

    elif mode == "Early Warnings":
        garch_p, garch_q, forecast_days = 1, 1, 14
        high_risk_vol, fear_threshold, greed_threshold = 60, 40, 60
        st.warning("Sensitive mode: Gives earlier risk alerts.")

    else:
        st.markdown(f"### {feather_icon('settings', 16)} Expert Controls", unsafe_allow_html=True)
        garch_p = st.slider("How much yesterday's panic matters", 1, 3, 1)
        garch_q = st.slider("How long market stress lasts", 1, 3, 1)
        forecast_days = st.slider("Days to predict future risk", 3, 14, 7)
        high_risk_vol = st.slider("Warn me when risk exceeds (%)", 50, 150, 80)
        fear_threshold = st.slider("When traders look scared", 10, 45, 30)
        greed_threshold = st.slider("When hype gets excessive", 55, 90, 70)

    st.markdown("---")
    st.markdown(f"### {feather_icon('wifi', 16)} Live Reddit Data", unsafe_allow_html=True)
    reddit_subs = st.multiselect(
        "Subreddits to fetch",
        ["CryptoCurrency", "Bitcoin", "ethereum", "solana", "CryptoMarkets"],
        default=["CryptoCurrency", "Bitcoin"],
    )
    reddit_sort = st.selectbox("Sort by", ["hot", "new", "top"], index=0)

    reddit_unlimited = st.toggle(
        "Fetch all available pages",
        value=False,
        help="Paginate through Reddit until no more posts exist. "
             "May take a minute — Reddit limits each page to 100 posts.",
    )
    if not reddit_unlimited:
        reddit_max = st.slider(
            "Max posts per subreddit", 25, 500, 100, step=25,
            help="Fetches multiple pages if needed to reach this number.",
        )
    else:
        reddit_max = None
        st.caption("Unlimited mode: will follow all pagination cursors. "
                   "Reddit typically exposes ~1,000 posts per subreddit.")

    fetch_btn = st.button("Fetch Live Reddit Posts", use_container_width=True)
    if fetch_btn:
        progress_bar  = st.progress(0, text="Starting fetch…")
        status_text   = st.empty()
        fetch_counter = {"n": 0}

        def on_progress(total, sub):
            fetch_counter["n"] = total
            cap = reddit_max if reddit_max else 1000
            pct = min(total / (cap * len(reddit_subs)), 1.0)
            progress_bar.progress(pct, text=f"Fetched {total} posts… (r/{sub})")
            status_text.caption(f"{total} posts collected so far")

        with st.spinner("Paginating through Reddit…"):
            result = fetch_reddit_posts(
                subreddits=reddit_subs,
                sort=reddit_sort,
                max_posts=reddit_max,
                progress_callback=on_progress,
                date_from=start_date,
                date_to=end_date,
            )

        progress_bar.empty()
        status_text.empty()

        if result:
            st.session_state.reddit_posts = result
            n = len([l for l in result.strip().split("\n") if l.strip()])
            st.success(f"Fetched {n} posts across {len(reddit_subs)} subreddit(s)!")
        else:
            st.error("Could not fetch Reddit posts. Using default data.")

    if st.session_state.reddit_posts:
        if st.button("Clear Live Data", use_container_width=True):
            st.session_state.reddit_posts = None
            st.rerun()
        st.caption("Live Reddit data active")

    st.markdown("---")
    run_btn = st.button("Run Analysis", use_container_width=True)




# ═══════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
<div class="hero">
    <p class="hero-title">{feather_icon('shield', 22)} Volasense</p>
    <p class="hero-sub">
        Crypto Fear &amp; Volatility Forecasting Engine &nbsp;·&nbsp;
        NLP Fear Index → GARCH(1,1) Volatility Model &nbsp;·&nbsp;
        Early Risk Detection System
    </p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Dashboard", "Social Media Input", "How It Works"])

# ── Tab 2: Input ──────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        f'<div class="section-hdr">{feather_icon("edit-3", 16)} Crypto Social Media Posts</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.reddit_posts:
        n_live = len([l for l in st.session_state.reddit_posts.strip().split("\n") if l.strip()])
        st.success(f"Live Reddit data loaded — {n_live} posts fetched. Editing below will override it.")
        active_posts = st.session_state.reddit_posts
    else:
        st.info("Tip: Click **Fetch Live Reddit Posts** in the sidebar to load real-time data, or edit the default dataset below.")
        active_posts = DEFAULT_POSTS

    st.markdown("""
    **Format:** `YYYY-MM-DD, post text here`
    Simulates Reddit (r/CryptoCurrency, r/Bitcoin) or Twitter/X posts.
    Each line = one post. The NLP engine scores each post and aggregates
    a daily **Fear & Greed Index** that feeds into the GARCH model.
    """)

    posts_input = st.text_area(
        "Posts", value=active_posts, height=450,
        label_visibility="collapsed",
    )

    n_posts = len([l for l in posts_input.strip().split("\n") if l.strip()])
    st.info(f"**{n_posts} posts** loaded across the date range.")

    st.markdown(
        f'<div class="section-hdr">{feather_icon("search", 16)} Live Post Tester</div>',
        unsafe_allow_html=True,
    )
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
    ## How Volasense Works

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
    | Danger Zone | Fear < 25 AND Vol > 80% | Panic selling, very high risk |
    | High Alert | Fear < 35 AND Vol > 60% | Market stressed, caution needed |
    | Euphoria Risk | Greed > 70 AND Vol > 90% | Over-extended, correction risk |
    | Bull Momentum | Greed > 65 AND Vol < 50% | Healthy uptrend, lower risk |

    ### References
    - Engle (1982). *Autoregressive Conditional Heteroskedasticity.* Econometrica.
    - Bollerslev (1986). *Generalized ARCH.* Journal of Econometrics.
    - Bollen et al. (2011). *Twitter mood predicts the stock market.* J. Comp. Science.
    - Nakano et al. (2018). *Bitcoin technical trading with NLP.* Finance Research Letters.
    """)

# ── Tab 1: Dashboard ──────────────────────────────────────────────────────
with tab1:
    if not run_btn and "crypto_results" not in st.session_state:
        st.markdown(f"""
        <div style="text-align:center;padding:80px 40px;color:#4b5563;">
            <div style="display:flex;justify-content:center;">
                <div style="width:84px;height:84px;color:#2563eb;">{feather_icon('shield', 84, 'fi-hero')}</div>
            </div>
            <div style="font-size:1.2rem;font-weight:700;color:#6b7280;margin-top:16px;">
                Configure settings in the sidebar and click
                <strong style="color:#2563eb">Run Analysis</strong>
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

        with st.spinner("Running NLP sentiment pipeline..."):
            posts_df  = analyze_posts(posts_input)
            daily_fear = aggregate_daily(posts_df)

        # If live Reddit data is used, dates are today's — auto-adjust
        # price range to match the actual post dates so they align
        if st.session_state.reddit_posts and not posts_df.empty:
            post_min = posts_df["date"].min().date()
            post_max = posts_df["date"].max().date()
            # Only override if posts are outside the user-selected range
            if post_min < start_date or post_max > end_date:
                start_date = post_min
                end_date   = post_max
                st.info(f"Date range auto-adjusted to match live Reddit posts: **{start_date}** → **{end_date}**")

        with st.spinner(f"Fetching {ticker} price data..."):
            price_df, real_data = fetch_crypto_prices(
                ticker, str(start_date), str(end_date)
            )

        with st.spinner("Fitting GARCH model..."):
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
            st.warning("Yahoo Finance unavailable — using synthetic GBM price data. NLP and GARCH are real.")

        if st.session_state.reddit_posts:
            n_live = len([l for l in st.session_state.reddit_posts.strip().split("\n") if l.strip()])
            st.success(f"Analysis powered by **{n_live} live Reddit posts**")

        # ── KPI Row ──────────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Live Dashboard</div>',
                    unsafe_allow_html=True)

        latest_fg    = merged["fear_greed"].dropna().iloc[-1]
        latest_vol   = merged["cond_vol"].dropna().iloc[-1]
        fc_max_vol   = forecast_df["volatility"].max()
        fc_risk      = forecast_df["risk_label"].iloc[0]
        persistence  = garch_metrics["persistence"]
        fg_label, fg_cls = classify_fear(latest_fg)
        n_danger = (merged["risk_zone"] == "Danger Zone").sum()

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

        st.markdown(
            f"<div class=\"section-hdr\">{feather_icon('trending-up', 16)} Today's Crypto Outlook</div>",
            unsafe_allow_html=True,
        )
        summary_parts = []
        if latest_fg < 25:
            summary_parts.append("Social media sentiment looks fearful.")
        elif latest_fg > 75:
            summary_parts.append("Social media sentiment looks bullish.")
        else:
            summary_parts.append("Social media sentiment looks mixed.")
        if fc_max_vol > 100:
            summary_parts.append("Large price swings are expected.")
        elif fc_max_vol < 50:
            summary_parts.append("Market conditions look relatively stable.")
        
        summary_text = "\n".join(summary_parts)
        summary_text += f"\n\nMarket mood: **{fg_label}**\n\nExpected risk: **{fc_risk}**"
        st.info(summary_text)


        # ── Charts ────────────────────────────────────────────────────────
        
        st.markdown(
            f'<div class="section-hdr">{feather_icon("bar-chart-2", 16)} Market Analysis Charts</div>',
            unsafe_allow_html=True,
        )

        chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5, chart_tab6 = st.tabs([
            "Price & Risk Level",
            "Market Sentiment",
            "Fear Impact",
            "Forecast",
            "Price Patterns",
            "Prediction vs Actual",
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

The teal line shows our prediction. The shaded band around it represents uncertainty — actual volatility will likely fall within this range.

Blue line shows recent history for context.
""")

        with chart_tab5:
            st.plotly_chart(chart_returns_dist(returns, cond_vol, st.session_state.theme), use_container_width=True)
            st.info("""
**How Do Price Swings Happen?**

**Left Chart: Distribution of daily returns**
Shows if movements are small & frequent or rare & extreme. The teal curve is what a 'normal' market would look like. Crypto usually has fatter tails (more extreme moves).

**Right Chart: Volatility clustering**
Notice how high-risk periods bunch together. One big move often triggers more big moves.
""")
            
        with chart_tab6:

            st.plotly_chart(
                chart_prediction_vs_actual(
                    returns,
                    cond_vol,
                    forecast_df,
                    st.session_state.theme
                ),
                use_container_width=True
            )

            st.info("""
        ### Prediction vs Actual Trend

        **Blue line**
        = Actual realized market volatility

        **Teal line**
        = GARCH predicted volatility

        **Amber line**
        = Future forecast

        If teal closely follows blue, the model predicts volatility well.
        This validates the computational model performance.
        """)

        # ── Forecast Table ────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-hdr">{feather_icon("table", 16)} Volatility Forecast Table</div>',
            unsafe_allow_html=True,
        )
        fc_display = forecast_df.copy().reset_index()
        fc_display["date"]       = fc_display["date"].dt.strftime("%Y-%m-%d")
        fc_display["volatility"] = fc_display["volatility"].map(lambda x: f"{x:.1f}%")
        fc_display = fc_display[["date","volatility","risk_label"]]
        fc_display.columns = ["Date","Forecasted Volatility","Risk Level"]
        st.dataframe(fc_display, use_container_width=True, hide_index=True)

        # ── GARCH Stats ───────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-hdr">{feather_icon("activity", 16)} GARCH Model Statistics</div>',
            unsafe_allow_html=True,
        )
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
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">{feather_icon('activity', 16)} How to read α+β (persistence)</div>
                <div class="insight-text">
                Close to <strong>1.0</strong> = volatility shocks last a long time (typical in crypto).<br>
                Close to <strong>0.5</strong> = shocks die out quickly.<br>
                Crypto typically shows 0.90–0.98 persistence.
                </div>
            </div>
            <div class="insight-box">
                <div class="insight-title">{feather_icon('bar-chart-2', 16)} NLP → GARCH connection</div>
                <div class="insight-text">
                The Fear & Greed Index (built from NLP) is used as a <strong>variance regressor</strong>
                in the GARCH model — high fear deviations directly increase
                the estimated conditional volatility, linking social sentiment
                to quantitative risk forecasting.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Top fearful / greedy posts ────────────────────────────────────
        st.markdown(
            f'<div class="section-hdr">{feather_icon("alert-triangle", 16)} Most Extreme Posts</div>',
            unsafe_allow_html=True,
        )
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
