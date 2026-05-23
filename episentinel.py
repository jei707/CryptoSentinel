"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EpiSentinel: NLP-Driven Epidemic Early Warning System               ║
║    Integrating Sentiment Analysis with SIR Epidemic + ARIMA Forecasting     ║
║                                                                              ║
║  Fields: Health Monitoring × Natural Language Processing                    ║
║  Models: VADER Sentiment Analysis | SIR Epidemic Model | ARIMA Forecasting  ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYSTEM OVERVIEW:
  1. NLP Layer     → VADER sentiment analysis on health-related social text
  2. Mapping Layer → Sentiment scores mapped to epidemic parameters (β, γ)
  3. SIR Model     → Epidemic simulation driven by sentiment-derived parameters
  4. ARIMA         → Time-series forecasting of infection trends
  5. Visualization → 5-panel dashboard of insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: SYNTHETIC DATASET
#   Simulates 90 days of health-related social media posts
#   (In production: replace with real Twitter/Reddit API data)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_health_social_data(n_days=90, seed=42):
    """
    Generate synthetic health-sentiment social media posts over 90 days.
    Three phases: calm → outbreak concern → recovery awareness.
    """
    np.random.seed(seed)

    phases = {
        "calm":     (0,  30,  "low concern"),
        "outbreak": (30, 65,  "rising panic"),
        "recovery": (65, 90,  "hopeful recovery"),
    }

    posts_per_day = np.random.randint(20, 60, n_days)

    calm_posts = [
        "Feeling a bit under the weather today, just a mild cold.",
        "Seasonal allergies acting up again, nothing serious.",
        "Heard there's a small flu going around the office.",
        "Kids had fever last night, seems fine now.",
        "Just a cough, probably from the weather change.",
        "Local clinic was busier than usual today, minor illnesses around.",
        "My neighbor mentioned a cold going around the block.",
        "Health department says everything is normal this season.",
        "Just a runny nose, nothing to worry about.",
        "Taking vitamins to stay healthy this winter.",
    ]

    outbreak_posts = [
        "Hospitals are OVERWHELMED. This disease is spreading fast! #sick",
        "Emergency rooms packed. Fever, cough, fatigue everywhere. SCARY.",
        "3 people from work are hospitalized. This is getting serious!!!",
        "Government confirms outbreak. EVERYONE STAY INSIDE. Very dangerous.",
        "My whole family is sick. This disease is NO JOKE. Stay safe people.",
        "Death toll rising. Health officials urging extreme caution. Frightening.",
        "Schools closing due to mass illness. This is a full epidemic.",
        "I'm terrified. So many people sick. When will this end?",
        "Can't get medicine anywhere. Pharmacies are all empty. Panic buying.",
        "This is getting out of control. Hospitals turning away patients.",
        "Lost someone to this outbreak. Heartbreaking. Please take this seriously.",
        "Health system collapsing. Need more doctors and beds immediately.",
    ]

    recovery_posts = [
        "New treatment is working! Cases starting to decline. Hopeful!",
        "Vaccine rollout beginning. Finally some good news!",
        "Hospitals reporting fewer new cases today. Recovery is real.",
        "Scientists say the worst is over. Community staying cautious but hopeful.",
        "My town reported zero new cases today! Amazing progress!",
        "People slowly returning to work. Life getting back to normal.",
        "Health officials optimistic. Measures are working, keep it up.",
        "Grateful for healthcare workers. Outbreak nearly under control.",
        "New cases at lowest level in weeks. Cautious optimism spreading.",
        "Community came together and beat this. Proud of everyone.",
    ]

    records = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    for day_idx in range(n_days):
        if day_idx < 30:
            pool = calm_posts
        elif day_idx < 65:
            pool = outbreak_posts
        else:
            pool = recovery_posts

        n = posts_per_day[day_idx]
        day_posts = np.random.choice(pool, size=n, replace=True)

        for post in day_posts:
            records.append({
                "date":  dates[day_idx],
                "day":   day_idx,
                "text":  post,
                "phase": "calm" if day_idx < 30 else ("outbreak" if day_idx < 65 else "recovery"),
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: NLP SENTIMENT ANALYSIS
#   Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
#   Optimized for short social media texts
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """
    VADER-based sentiment analyzer tailored for health-crisis monitoring.
    Computes compound sentiment scores and maps them to health-risk levels.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

        # Health-crisis domain keywords (boosts/dampens scores)
        self.crisis_boosters = {
            "overwhelmed": 0.4, "terrified": 0.5, "hospitalized": 0.3,
            "outbreak": 0.3, "epidemic": 0.4, "pandemic": 0.5,
            "dying": 0.5, "death": 0.4, "collapsing": 0.4,
            "dangerous": 0.3, "scary": 0.3, "panic": 0.4,
        }
        self.recovery_boosters = {
            "vaccine": 0.3, "treatment": 0.2, "recovering": 0.2,
            "hopeful": 0.3, "declining": 0.2, "improving": 0.3,
        }

    def analyze(self, text):
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Domain-aware adjustment
        text_lower = text.lower()
        adjustment = 0.0
        for kw, boost in self.crisis_boosters.items():
            if kw in text_lower:
                adjustment -= boost  # Crisis language → more negative
        for kw, boost in self.recovery_boosters.items():
            if kw in text_lower:
                adjustment += boost  # Recovery language → more positive

        adjusted = np.clip(compound + adjustment * 0.3, -1.0, 1.0)
        return {
            "compound":   compound,
            "adjusted":   adjusted,
            "positive":   scores["pos"],
            "negative":   scores["neg"],
            "neutral":    scores["neu"],
            "risk_level": self._risk_level(adjusted),
        }

    def _risk_level(self, score):
        if score >= 0.3:   return "Low"
        elif score >= 0.0: return "Moderate"
        elif score >= -0.4:return "High"
        else:              return "Critical"

    def analyze_dataframe(self, df):
        results = df["text"].apply(self.analyze)
        sentiment_df = pd.DataFrame(results.tolist())
        return pd.concat([df, sentiment_df], axis=1)

    def aggregate_daily(self, df):
        return df.groupby("day").agg(
            date=("date", "first"),
            phase=("phase", "first"),
            mean_compound=("compound",  "mean"),
            mean_adjusted=("adjusted",  "mean"),
            mean_positive=("positive",  "mean"),
            mean_negative=("negative",  "mean"),
            mean_neutral=("neutral",    "mean"),
            post_count=("text",         "count"),
            risk_pct_critical=("risk_level", lambda x: (x == "Critical").mean()),
            risk_pct_high=("risk_level",     lambda x: (x == "High").mean()),
        ).reset_index()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: SIR EPIDEMIC MODEL (Computational Science)
#   S = Susceptible, I = Infected, R = Recovered
#   Parameters β (transmission) and γ (recovery) are SENTIMENT-DRIVEN
# ═══════════════════════════════════════════════════════════════════════════════

class SIREpidemicModel:
    """
    SIR (Susceptible-Infected-Recovered) epidemic model.
    The key innovation: β (transmission rate) is dynamically modulated
    by daily public sentiment — negative sentiment → higher β (fear spreads,
    people avoid precautions or panic-crowd hospitals), positive sentiment
    → lower β (people follow guidelines, maintain social distancing).
    """

    def __init__(self, N=100_000):
        self.N = N  # Total population

    def sentiment_to_beta(self, sentiment_score):
        """
        Map daily sentiment [-1, +1] to transmission rate β [0.05, 0.45].
        Negative sentiment → higher β (panic, distrust, less compliance).
        Positive sentiment → lower β (trust, compliance, precaution).
        """
        # Invert: more negative sentiment = higher transmission
        normalized = (-sentiment_score + 1) / 2   # map [-1,1] → [1,0]
        beta = 0.05 + normalized * 0.40
        return float(np.clip(beta, 0.05, 0.45))

    def sentiment_to_gamma(self, sentiment_score):
        """
        Map daily sentiment to recovery rate γ [0.03, 0.15].
        Positive sentiment → higher γ (better healthcare cooperation, morale).
        """
        normalized = (sentiment_score + 1) / 2    # map [-1,1] → [0,1]
        gamma = 0.03 + normalized * 0.12
        return float(np.clip(gamma, 0.03, 0.15))

    def _sir_equations(self, y, t, beta, gamma):
        S, I, R = y
        N = self.N
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    def simulate(self, daily_sentiments, I0=100):
        """
        Simulate SIR dynamics day-by-day using sentiment-driven parameters.
        Returns trajectory of S, I, R, β, γ, and R₀ for each day.
        """
        S, I, R = self.N - I0, I0, 0
        n_days = len(daily_sentiments)

        results = []
        for day, sentiment in enumerate(daily_sentiments):
            beta  = self.sentiment_to_beta(sentiment)
            gamma = self.sentiment_to_gamma(sentiment)
            R0    = beta / gamma

            t = np.linspace(0, 1, 10)
            sol = odeint(self._sir_equations, [S, I, R], t,
                         args=(beta, gamma), full_output=False)
            S, I, R = sol[-1]
            S, I, R = max(0, S), max(0, I), max(0, R)

            results.append({
                "day":       day,
                "S":         S,
                "I":         I,
                "R":         R,
                "beta":      beta,
                "gamma":     gamma,
                "R0":        R0,
                "sentiment": sentiment,
            })

        return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: ARIMA FORECASTING
#   Forecasts future infection counts from historical I(t) trajectory
# ═══════════════════════════════════════════════════════════════════════════════

class ARIMAForecaster:
    """
    ARIMA time-series model to forecast infection trajectory.
    Trained on first 70% of SIR-simulated infection data;
    forecasts remaining 30% for validation.
    """

    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.model  = None
        self.result = None

    def fit_and_forecast(self, series, forecast_steps):
        try:
            self.model  = ARIMA(series, order=self.order)
            self.result = self.model.fit()
            forecast    = self.result.forecast(steps=forecast_steps)
            ci          = self.result.get_forecast(steps=forecast_steps).conf_int()
            return np.maximum(forecast.values, 0), ci
        except Exception:
            # Fallback: simple moving average if ARIMA fails
            window   = min(7, len(series))
            trend    = np.mean(series[-window:])
            forecast = np.full(forecast_steps, trend)
            lower    = forecast * 0.85
            upper    = forecast * 1.15
            ci = pd.DataFrame({"lower I": lower, "upper I": upper})
            return forecast, ci

    def compute_metrics(self, actual, predicted):
        n   = min(len(actual), len(predicted))
        mae = mean_absolute_error(actual[:n], predicted[:n])
        rmse= np.sqrt(mean_squared_error(actual[:n], predicted[:n]))
        # MAPE with zero guard — use threshold > 500 to avoid near-zero distortion
        mask = actual[:n] > 500
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[:n][mask] - predicted[:n][mask])
                                  / actual[:n][mask])) * 100
        else:
            mape = float("nan")
        return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: VISUALIZATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(daily_agg, sir_df, actual_I, forecast_I, ci,
                   split_day, metrics):
    """
    5-panel dashboard:
      Panel 1: Sentiment trend over time (NLP output)
      Panel 2: Risk-level distribution over time (NLP output)
      Panel 3: SIR epidemic simulation (computational model output)
      Panel 4: ARIMA forecast vs actual (prediction model output)
      Panel 5: β / R₀ driven by sentiment (NLP→model linkage)
    """
    # ── Color palette ─────────────────────────────────────────────────────────
    DARK   = "#0d1117"
    CARD   = "#161b22"
    ACCENT = "#58a6ff"
    GREEN  = "#3fb950"
    RED    = "#f85149"
    ORANGE = "#d29922"
    PURPLE = "#bc8cff"
    GRAY   = "#8b949e"
    WHITE  = "#e6edf3"

    phase_colors = {"calm": GREEN, "outbreak": RED, "recovery": ACCENT}

    fig = plt.figure(figsize=(20, 22), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.32,
                           top=0.91, bottom=0.05,
                           left=0.07, right=0.96)

    ax1 = fig.add_subplot(gs[0, 0])  # Sentiment trend
    ax2 = fig.add_subplot(gs[0, 1])  # Risk distribution
    ax3 = fig.add_subplot(gs[1, :])  # SIR model (full width)
    ax4 = fig.add_subplot(gs[2, 0])  # ARIMA forecast
    ax5 = fig.add_subplot(gs[2, 1])  # β and R₀ over time

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=GRAY, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    days = daily_agg["day"].values

    # ── Panel 1: Sentiment Trend ───────────────────────────────────────────────
    ax1.set_title("① Sentiment Trend Over Time  [NLP Output]",
                  fontsize=11, fontweight="bold", pad=8)

    # Shade phases
    phase_spans = [(0, 30, "Calm", GREEN), (30, 65, "Outbreak", RED),
                   (65, 90, "Recovery", ACCENT)]
    for start, end, label, color in phase_spans:
        ax1.axvspan(start, end, alpha=0.08, color=color, label=label)
        ax1.text((start + end) / 2, 0.85, label,
                 ha="center", va="top", fontsize=8,
                 color=color, transform=ax1.get_xaxis_transform(), alpha=0.8)

    ax1.plot(days, daily_agg["mean_adjusted"], color=ACCENT,
             lw=2, label="Adjusted Sentiment", zorder=3)
    ax1.fill_between(days, daily_agg["mean_adjusted"], 0,
                     where=daily_agg["mean_adjusted"] < 0,
                     alpha=0.3, color=RED, label="Negative Zone")
    ax1.fill_between(days, daily_agg["mean_adjusted"], 0,
                     where=daily_agg["mean_adjusted"] >= 0,
                     alpha=0.2, color=GREEN, label="Positive Zone")
    ax1.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Sentiment Score")
    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(fontsize=7, loc="lower right",
               facecolor=DARK, labelcolor=WHITE, edgecolor=GRAY)
    ax1.grid(axis="y", alpha=0.15, color=GRAY)

    # ── Panel 2: Risk Distribution Stacked Area ────────────────────────────────
    ax2.set_title("② Public Risk-Level Distribution  [NLP Output]",
                  fontsize=11, fontweight="bold", pad=8)

    pct_critical = daily_agg["risk_pct_critical"].values * 100
    pct_high     = daily_agg["risk_pct_high"].values * 100
    pct_low      = np.clip(100 - pct_critical - pct_high, 0, 100)

    ax2.stackplot(days,
                  pct_low, pct_high, pct_critical,
                  labels=["Low/Moderate", "High", "Critical"],
                  colors=[GREEN + "bb", ORANGE + "bb", RED + "bb"],
                  alpha=0.85)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("% of Posts")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=7, loc="upper left",
               facecolor=DARK, labelcolor=WHITE, edgecolor=GRAY)
    ax2.grid(axis="y", alpha=0.15, color=GRAY)

    # ── Panel 3: SIR Simulation ────────────────────────────────────────────────
    ax3.set_title(
        "③ SIR Epidemic Model — Sentiment-Driven Parameters  [Computational Science]",
        fontsize=11, fontweight="bold", pad=8)

    ax3.plot(sir_df["day"], sir_df["S"] / 1000, color=ACCENT,
             lw=2.5, label="Susceptible (S)")
    ax3.plot(sir_df["day"], sir_df["I"] / 1000, color=RED,
             lw=2.5, label="Infected (I)", zorder=4)
    ax3.fill_between(sir_df["day"], sir_df["I"] / 1000,
                     alpha=0.15, color=RED)
    ax3.plot(sir_df["day"], sir_df["R"] / 1000, color=GREEN,
             lw=2.5, label="Recovered (R)")

    # Annotation: peak infection
    peak_idx = sir_df["I"].idxmax()
    peak_day = sir_df.loc[peak_idx, "day"]
    peak_I   = sir_df.loc[peak_idx, "I"] / 1000
    ax3.annotate(f"Peak: {peak_I:,.0f}K\nDay {peak_day}",
                 xy=(peak_day, peak_I),
                 xytext=(peak_day + 5, peak_I + 2),
                 arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.2),
                 color=WHITE, fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc=CARD, ec=RED, alpha=0.9))

    # Phase shading
    for start, end, label, color in phase_spans:
        ax3.axvspan(start, end, alpha=0.05, color=color)

    ax3.set_xlabel("Day")
    ax3.set_ylabel("Population (thousands)")
    ax3.legend(fontsize=9, loc="center right",
               facecolor=DARK, labelcolor=WHITE, edgecolor=GRAY)
    ax3.grid(alpha=0.12, color=GRAY)

    # ── Panel 4: ARIMA Forecast vs Actual ─────────────────────────────────────
    ax4.set_title("④ ARIMA Forecast vs Actual Infections  [Prediction Model]",
                  fontsize=11, fontweight="bold", pad=8)

    train_days    = np.arange(split_day)
    forecast_days = np.arange(split_day, split_day + len(forecast_I))

    ax4.plot(train_days, actual_I[:split_day] / 1000,
             color=ACCENT, lw=2, label="Actual (Training)")
    ax4.plot(forecast_days, actual_I[split_day:split_day + len(forecast_I)] / 1000,
             color=GREEN, lw=2, label="Actual (Test)", ls="--")
    ax4.plot(forecast_days, forecast_I / 1000,
             color=ORANGE, lw=2, label="ARIMA Forecast", ls="-.")

    if ci is not None:
        try:
            ci_lower = np.maximum(ci.iloc[:, 0].values, 0) / 1000
            ci_upper = ci.iloc[:, 1].values / 1000
            ax4.fill_between(forecast_days, ci_lower, ci_upper,
                             alpha=0.2, color=ORANGE, label="95% CI")
        except Exception:
            pass

    ax4.axvline(split_day, color=GRAY, lw=1, ls=":", alpha=0.6)
    ax4.text(split_day + 0.5, ax4.get_ylim()[1] * 0.9, "Forecast\nStart",
             color=GRAY, fontsize=7)

    # Metrics box
    metric_text = (f"MAE:  {metrics['MAE']:,.0f}\n"
                   f"RMSE: {metrics['RMSE']:,.0f}\n"
                   f"MAPE: {metrics['MAPE (%)']:.1f}%")
    ax4.text(0.02, 0.97, metric_text, transform=ax4.transAxes,
             fontsize=8, color=WHITE, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc=DARK, ec=ACCENT, alpha=0.85),
             fontfamily="monospace")

    ax4.set_xlabel("Day")
    ax4.set_ylabel("Infected (thousands)")
    ax4.legend(fontsize=7, loc="upper right",
               facecolor=DARK, labelcolor=WHITE, edgecolor=GRAY)
    ax4.grid(alpha=0.12, color=GRAY)

    # ── Panel 5: β & R₀ over time (NLP → Model linkage) ──────────────────────
    ax5.set_title("⑤ Sentiment → β (Transmission) & R₀  [NLP–Model Bridge]",
                  fontsize=11, fontweight="bold", pad=8)

    ax5_r = ax5.twinx()
    ax5_r.tick_params(colors=GRAY, labelsize=9)
    ax5_r.yaxis.label.set_color(WHITE)

    l1, = ax5.plot(sir_df["day"], sir_df["beta"],
                   color=RED, lw=2, label="β (transmission rate)")
    l2, = ax5_r.plot(sir_df["day"], sir_df["R0"],
                     color=ORANGE, lw=2, ls="--", label="R₀ (reproduction number)")
    ax5_r.axhline(1.0, color=GRAY, lw=0.8, ls=":", alpha=0.7)
    ax5_r.text(2, 1.05, "R₀ = 1 (epidemic threshold)",
               color=GRAY, fontsize=7, alpha=0.8)

    for spine in ax5_r.spines.values():
        spine.set_edgecolor("#30363d")

    ax5.set_xlabel("Day")
    ax5.set_ylabel("β  (transmission rate)", color=RED)
    ax5_r.set_ylabel("R₀  (basic reproduction number)", color=ORANGE)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, fontsize=7, loc="upper right",
               facecolor=DARK, labelcolor=WHITE, edgecolor=GRAY)
    ax5.grid(alpha=0.12, color=GRAY)

    # ── Main title ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.955,
             "EpiSentinel: NLP-Driven Epidemic Early Warning System",
             ha="center", va="center", fontsize=16, fontweight="bold",
             color=WHITE)
    fig.text(0.5, 0.935,
             "Integrating VADER Sentiment Analysis  ×  SIR Epidemic Model  ×  ARIMA Forecasting",
             ha="center", va="center", fontsize=10, color=GRAY)

    plt.savefig("episentinel_dashboard.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=DARK)
    print("  ✓  Dashboard saved → episentinel_dashboard.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("  EpiSentinel — NLP × Computational Science Integration System")
    print("═" * 70 + "\n")

    # ── Step 1: Generate / Load Data ──────────────────────────────────────────
    print("► Step 1: Generating synthetic health-sentiment dataset …")
    df = generate_health_social_data(n_days=90)
    print(f"  Total posts: {len(df):,}  |  Days: 90  |  "
          f"Phases: calm (d0-29), outbreak (d30-64), recovery (d65-89)")

    # ── Step 2: NLP Sentiment Analysis ────────────────────────────────────────
    print("\n► Step 2: Running VADER Sentiment Analysis …")
    sa = SentimentAnalyzer()
    df = sa.analyze_dataframe(df)
    daily_agg = sa.aggregate_daily(df)

    print(f"  Mean sentiment by phase:")
    for phase in ["calm", "outbreak", "recovery"]:
        s = daily_agg[daily_agg["phase"] == phase]["mean_adjusted"].mean()
        bar = "▓" * int(abs(s) * 20)
        sign = "+" if s >= 0 else "-"
        print(f"    {phase:10s}  {sign}{abs(s):.3f}  {bar}")

    # ── Step 3: SIR Epidemic Simulation ───────────────────────────────────────
    print("\n► Step 3: Running SIR Epidemic Model (sentiment-driven β, γ) …")
    sir_model  = SIREpidemicModel(N=100_000)
    sentiments = daily_agg["mean_adjusted"].values
    sir_df     = sir_model.simulate(sentiments, I0=50)

    peak_I   = sir_df["I"].max()
    peak_day = sir_df.loc[sir_df["I"].idxmax(), "day"]
    total_R  = sir_df["R"].iloc[-1]
    print(f"  Peak infections: {peak_I:,.0f}  on day {peak_day}")
    print(f"  Total recovered: {total_R:,.0f}  by day 90")
    print(f"  Max R₀:          {sir_df['R0'].max():.2f}")

    # ── Step 4: ARIMA Forecasting ──────────────────────────────────────────────
    print("\n► Step 4: Fitting ARIMA(2,1,2) — forecasting infection trajectory …")
    split = 40  # split before peak — forecasts the most critical outbreak phase
    I_series    = sir_df["I"].values
    train_series= I_series[:split]
    n_forecast  = len(I_series) - split

    arima = ARIMAForecaster(order=(2, 1, 2))
    forecast_I, ci = arima.fit_and_forecast(train_series, n_forecast)
    metrics = arima.compute_metrics(I_series[split:], forecast_I)

    print(f"  Training days: {split}  |  Forecast days: {n_forecast}")
    print(f"  MAE:  {metrics['MAE']:>10,.1f} people")
    print(f"  RMSE: {metrics['RMSE']:>10,.1f} people")
    print(f"  MAPE: {metrics['MAPE (%)']:>9.2f}%")

    # ── Step 5: Visualize ──────────────────────────────────────────────────────
    print("\n► Step 5: Generating 5-panel visualization dashboard …")
    plot_dashboard(daily_agg, sir_df, I_series, forecast_I, ci, split, metrics)

    # ── Summary Report ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  SYSTEM SUMMARY REPORT")
    print("─" * 70)
    print(f"  NLP Engine       :  VADER Sentiment Analyzer (domain-boosted)")
    print(f"  Comp. Model 1    :  SIR Epidemic Model (ODE-based, N=100,000)")
    print(f"  Comp. Model 2    :  ARIMA(2,1,2) Time-Series Forecasting")
    print(f"  Key Integration  :  Sentiment score → β (transmission rate)")
    print(f"  Total text posts :  {len(df):,}")
    print(f"  Peak infection   :  {peak_I:,.0f} people (day {peak_day})")
    print(f"  Attack rate      :  {total_R / 100_000 * 100:.1f}% of population")
    print(f"  Forecast MAPE    :  {metrics['MAPE (%)']:.2f}%")
    print("─" * 70)
    print("\n  ✓  EpiSentinel pipeline complete.\n")

    return df, daily_agg, sir_df, forecast_I, metrics

# ==========================
# STREAMLIT APP
# ==========================

st.set_page_config(

    page_title="EpiSentinel",

    layout="wide"

)

st.title(
    "🦠 EpiSentinel"
)

st.subheader(
    "NLP-Driven Epidemic Early Warning System"
)


source = st.radio(

    "Choose data source",

    [

        "Generated Data",

        "Upload CSV"

    ]

)


uploaded = None

if source=="Upload CSV":

    uploaded = st.file_uploader(

        "Upload CSV",

        type=["csv"]

    )


if st.button(

    "Run EpiSentinel"

):

    # -------------------------
    # DATA
    # -------------------------

    if uploaded:

        df = pd.read_csv(

            uploaded

        )

        df["date"] = pd.to_datetime(

            df["date"]

        )

        df["day"] = (

            df["date"]

            -

            df["date"].min()

        ).dt.days

        df["phase"]="uploaded"


    else:

        df = generate_health_social_data(

            n_days=90

        )


    # -------------------------
    # NLP
    # -------------------------

    sa = SentimentAnalyzer()

    df = sa.analyze_dataframe(

        df

    )

    daily_agg = sa.aggregate_daily(

        df

    )


    # -------------------------
    # SIR
    # -------------------------

    sir_model = SIREpidemicModel(

        N=100000

    )

    sir_df = sir_model.simulate(

        daily_agg["mean_adjusted"]

    )


    # -------------------------
    # ARIMA
    # -------------------------

    split = 40

    I = sir_df["I"].values


    arima = ARIMAForecaster()


    forecast, ci = arima.fit_and_forecast(

        I[:split],

        len(I)-split

    )


    metrics = arima.compute_metrics(

        I[split:],

        forecast

    )


    # =========================
    # METRIC CARDS
    # =========================


    c1,c2,c3,c4 = st.columns(

        4

    )


    c1.metric(

        "Peak Infection",

        f"{int(sir_df['I'].max()):,}"

    )


    c2.metric(

        "Mean Sentiment",

        f"{daily_agg['mean_adjusted'].mean():.2f}"

    )


    c3.metric(

        "Forecast RMSE",

        f"{metrics['RMSE']:.0f}"

    )


    c4.metric(

        "Max R₀",

        f"{sir_df['R0'].max():.2f}"

    )


    # ======================
    # ALERT
    # ======================


    if sir_df["R0"].max()>1:

        st.error(

            "⚠ Epidemic Risk Increasing"

        )

    else:

        st.success(

            "✓ Risk Controlled"

        )


    tabs = st.tabs(

        [

            "Sentiment",

            "Infections",

            "Forecast",

            "Risk"

        ]

    )


    # =====================
    # TAB 1
    # =====================

    with tabs[0]:

        fig = px.line(

            daily_agg,

            x="day",

            y="mean_adjusted",

            title="Sentiment Trend"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )


    # =====================
    # TAB 2
    # =====================

    with tabs[1]:

        fig = px.line(

            sir_df,

            x="day",

            y=["S","I","R"],

            title="SIR Simulation"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )


    # =====================
    # TAB 3
    # =====================


    with tabs[2]:


        forecast_days = np.arange(

            split,

            split+len(forecast)

        )


        fig = go.Figure()


        fig.add_trace(

            go.Scatter(

                x=np.arange(

                    len(I)

                ),

                y=I,

                name="Actual"

            )

        )


        fig.add_trace(

            go.Scatter(

                x=forecast_days,

                y=forecast,

                name="Forecast"

            )

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


    # =====================
    # TAB 4
    # =====================


with tabs[3]:

    risk = abs(
        daily_agg["mean_adjusted"].mean()
    ) * 100


    # Change color based on risk level
    if risk < 25:
        num_color = "#00C853"      # Green

    elif risk < 50:
        num_color = "#FFD600"      # Yellow

    elif risk < 75:
        num_color = "#FF6D00"      # Orange

    else:
        num_color = "#D50000"      # Red


    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=risk,

            title={

                'text':

                "Epidemic Risk"

            },

            number={

                'font': {

                    'size':60,

                    'color': num_color

                }

            },

            gauge={

                'axis': {

                    'range':[0,100]

                },


                # Needle/bar color changes too
                'bar': {

                    'color': num_color

                },


                'steps':[

                    {

                        'range':[0,25],

                        'color':"#00C853"

                    },

                    {

                        'range':[25,50],

                        'color':"#FFD600"

                    },

                    {

                        'range':[50,75],

                        'color':"#FF6D00"

                    },

                    {

                        'range':[75,100],

                        'color':"#D50000"

                    }

                ]

            }

        )

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )