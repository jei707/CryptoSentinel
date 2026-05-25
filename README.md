# Volasense

Volasense is an interactive Streamlit application for crypto risk monitoring and volatility forecasting.
It converts social-media text into a daily Fear and Greed Index using NLP (VADER plus a small crypto lexicon), then fits a GARCH volatility model to produce an annualized volatility series and a forward volatility forecast with a simple risk classification.

This repo contains a single-file Streamlit app located in `main.py`.

## What it does

Volasense combines three things:

1. NLP sentiment scoring on posts
   - Ingests a dataset of dated posts.
   - Scores each post with VADER sentiment.
   - Adjusts the score with a small crypto-specific lexicon.
   - Maps the final compound sentiment value to a 0 to 100 Fear and Greed Index.

2. Market data ingestion
   - Pulls price and volume data from Yahoo Finance using `yfinance`.
   - If Yahoo Finance is unavailable or returns empty data, the app falls back to a synthetic price series (GBM with regime shifts) so the rest of the pipeline can still run.

3. Volatility modeling and forecasting
   - Computes daily percent returns from prices.
   - Fits a GARCH(p,q) model using the `arch` package.
   - Produces conditional volatility (annualized) and a multi-day forecast.
   - Classifies forecast days into simple risk buckets.

## UI overview

The app has:

- Sidebar configuration
  - Theme: dark or light (CSS plus chart theme).
  - Crypto asset: choose a ticker (BTC-USD, ETH-USD, BNB-USD, SOL-USD, or a custom ticker).
  - Date range: start and end dates.
  - Analysis mode:
      - Beginner, Balanced, Early Warnings, or Advanced.
      - Advanced exposes sliders for GARCH p/q, forecast horizon, and some thresholds.
  - Live Reddit data:
      - Optional fetch of public Reddit JSON posts from selected subreddits.
      - Supports a bounded mode (max posts per subreddit) and an unlimited pagination mode.

- Main tabs
  - Dashboard: KPIs, summary text, and charts.
  - Social Media Input: editable dataset plus a single-post sentiment tester.
  - How It Works: explanation of the pipeline and the model.

## How the pipeline works

High-level data flow:

1. Posts dataset (date, text)
2. VADER sentiment per post
3. Daily aggregation to Fear and Greed Index
4. Price download for chosen ticker and date range
5. Returns computation
6. GARCH model fit
7. Conditional volatility plus forecast
8. Risk classification and visualizations

Notes about the implementation:

- The app attempts to fit a GARCH model that includes the Fear and Greed series as an external regressor (variance regressor). If that fit fails, it falls back to the baseline GARCH fit.
- Conditional volatility and forecasts in the current implementation are computed from the baseline GARCH fit.
- The UI reports whether the regressor fit succeeded via a `used_fear_regressor` flag in the computed metrics.

## Configuration reference

The sidebar controls map directly to parameters used by the NLP and GARCH pipeline.

### Theme

- Dark Mode or Light Mode.
- The theme triggers a rerun and updates both CSS and Plotly chart colors.

### Crypto asset

- Presets: BTC-USD, ETH-USD, BNB-USD, SOL-USD.
- Custom: free-text ticker (for example DOGE-USD).

### Date range

- Start and End date selectors.
- If live Reddit data is enabled, the app may auto-adjust the price date range so price data aligns with the actual post dates.

### Analysis mode

- Beginner (Recommended)
  - GARCH p=1, q=1
  - Forecast horizon: 7 business days
  - Threshold defaults: high_risk_vol=80, fear_threshold=30, greed_threshold=70

- Balanced
  - GARCH p=1, q=1
  - Forecast horizon: 10 business days
  - Threshold defaults: high_risk_vol=70, fear_threshold=35, greed_threshold=65

- Early Warnings
  - GARCH p=1, q=1
  - Forecast horizon: 14 business days
  - Threshold defaults: high_risk_vol=60, fear_threshold=40, greed_threshold=60

- Advanced
  - Sliders are exposed for:
    - GARCH p (range 1 to 3)
    - GARCH q (range 1 to 3)
    - Forecast horizon in days (range 3 to 14)
    - Warning volatility threshold (range 50 to 150)
    - Fear threshold (range 10 to 45)
    - Greed threshold (range 55 to 90)

### Live Reddit data

- Subreddits: multi-select list.
- Sort: hot, new, or top.
- Bounded mode: sets a maximum posts per subreddit (25 to 500, step 25).
- Unlimited mode: follows pagination cursors until no more posts exist. This can be slower and may hit rate limits.

## Requirements

- Python (a recent 3.x version is recommended)
- Internet access (optional; needed for live Reddit fetch and Yahoo Finance price download)

Python packages:

```bash
pip install -r requirements.txt
```

If you prefer to install manually:

```bash
pip install streamlit pandas numpy plotly requests vaderSentiment arch yfinance
```

## Quick start

From this directory:

```bash
streamlit run main.py
```

If you prefer a virtual environment (recommended):

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

## Using the Social Media dataset

The Social Media Input tab expects a simple CSV-like text format:

```
YYYY-MM-DD, post text here
YYYY-MM-DD, another post
```

Guidelines:

- One post per line.
- The first comma separates the date from the post text.
- The date is parsed and used for daily aggregation.

The repo ships with a built-in default dataset in `DEFAULT_POSTS` inside `main.py`.

## Live Reddit fetch details

The live fetch uses Reddit's public JSON endpoints (no API keys) and a custom User-Agent string.

Important limitations:

- Reddit may rate-limit requests (HTTP 429). The app performs a small backoff and retries once.
- Public endpoints can change and may occasionally return unexpected data.
- Unlimited pagination can take time; expect the fetch to be slower when you select more subreddits or request more pages.

## Forecast and risk labels

Volatility is reported as annualized percent.
The forward forecast is classified into these buckets:

- Low Risk
- Moderate Risk
- High Risk
- Extreme Risk

These labels are based on the forecast volatility thresholds inside `fit_garch`.

Separately, the dashboard also computes a simple combined risk-zone label based on the Fear and Greed value and conditional volatility:

- Danger Zone
- High Alert
- Euphoria Risk
- Bull Momentum
- Normal

## Charts and outputs

Dashboard outputs include:

- KPI cards for recent Fear and Greed, conditional volatility, forecast peak volatility and risk label, persistence (alpha plus beta), posts analyzed, and danger-day counts.

The dashboard charts include:

1. Price vs Market Risk
   - Top panel shows price.
   - Bottom panel shows conditional volatility.
   - Background bands represent risk-zone labels.

2. Social sentiment and volume
   - Daily Fear and Greed (raw and smoothed).
   - Post count over time.

3. Fear impact scatter
   - Scatter of Fear and Greed versus conditional volatility.
   - A fitted trend line is shown when there is sufficient data.

4. Volatility forecast
   - Recent historical volatility plus forecast line.
   - Uncertainty band around the forecast.

5. Returns distribution and clustering
   - Histogram of returns with an overlaid normal reference.
   - Conditional volatility series to illustrate clustering.

6. Prediction vs Actual
   - Compares realized rolling volatility, modeled volatility, and the forward forecast.

Additional tables on the dashboard:

- Volatility forecast table (date, forecast volatility, risk label)
- GARCH model statistics (AIC, BIC, log-likelihood, parameters, persistence)
- Most extreme posts (most fearful and most greedy by sentiment score)

## Theme and styling

The UI theme is implemented via a CSS injection function `get_theme_css(theme)`.
The application uses inline Feather-style SVG icons (embedded directly in HTML) for section headings and some label surfaces.

## Project structure

This project is intentionally small:

```
CryptoSentinel/
  main.py
   requirements.txt
  README.md
```

`main.py` contains:

- Streamlit UI
- CSS theming
- Reddit fetch
- NLP scoring and aggregation
- Price download and synthetic fallback
- GARCH fit and forecast
- Plotly charts

## Troubleshooting

Common issues:

1. Import errors in your editor
   - Ensure the same Python environment used by your editor has the dependencies installed.

2. Yahoo Finance returns empty data
   - This can happen due to rate limits, ticker typos, or intermittent upstream issues.
   - The app will fall back to synthetic data so you can still explore the pipeline.

3. Reddit fetch fails
   - Check your network.
   - Reduce the number of subreddits or disable unlimited pagination.
   - Try again later if you are being rate-limited.

## Disclaimer

This project is for educational and research purposes. It is not financial advice.


