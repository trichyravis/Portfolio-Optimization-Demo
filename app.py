
"""
Dynamic Efficient Frontier — Multi-Asset Portfolio Optimizer
The Mountain Path - World of Finance
Prof. V. Ravichandran
28+ Years Corporate Finance & Banking Experience | 10+ Years Academic Excellence

Single-file Streamlit application implementing Modern Portfolio Theory:
    - Real-time data from Yahoo Finance
    - Monte Carlo simulation (Dirichlet weights)
    - Mean-Variance Optimization (SLSQP)
    - Efficient Frontier curve tracing
    - Capital Market Line (CML)
    - Max Sharpe / Min Variance / Risk Parity portfolios
    - Correlation & Covariance heatmaps
    - Normalized price history & drawdown analysis
    - CSV export
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

APP_TITLE = "Dynamic Efficient Frontier | Multi-Asset Portfolio Optimizer"
APP_ICON = "📈"
AUTHOR = "Prof. V. Ravichandran"
AUTHOR_CREDENTIALS = "28+ Years Corporate Finance & Banking Experience | 10+ Years Academic Excellence"
PLATFORM = "The Mountain Path - World of Finance"
LINKEDIN_URL = "https://www.linkedin.com/in/trichyravis"
GITHUB_URL = "https://github.com/trichyravis"

COLORS = {
    "gold": "#FFD700", "dark_blue": "#003366", "mid_blue": "#004d80",
    "card_bg": "#112240", "bg_primary": "#0a192f", "text_primary": "#e6f1ff",
    "text_muted": "#8892b0", "green": "#28a745", "red": "#dc3545",
    "light_blue": "#ADD8E6", "cyan": "#64ffda",
}

PLOTLY_COLORS = [
    "#FFD700", "#64ffda", "#ff6b6b", "#4ecdc4", "#45b7d1",
    "#96ceb4", "#ffeaa7", "#dfe6e9", "#fd79a8", "#a29bfe",
    "#00b894", "#e17055", "#0984e3", "#6c5ce7", "#fdcb6e",
]

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"]
DEFAULT_NUM_PORTFOLIOS = 10000
DEFAULT_RISK_FREE_RATE = 0.07
DEFAULT_TRADING_DAYS = 252


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: CSS STYLES
# ═════════════════════════════════════════════════════════════════════════════

PAGE_CSS = f"""
<style>
    .stApp {{ background: linear-gradient(135deg, #1a2332, #243447, #2a3f5f); }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
        border-right: 1px solid {COLORS['gold']}33;
    }}
    section[data-testid="stSidebar"] .stMarkdown p {{ color: {COLORS['text_primary']}; }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{ color: {COLORS['gold']} !important; }}

    /* Sidebar widget labels */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stTextArea label,
    section[data-testid="stSidebar"] .stDateInput label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {{
        color: {COLORS['text_primary']} !important;
    }}

    /* Sidebar slider value, help text, small text */
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] span {{
        color: {COLORS['text_muted']} !important;
    }}

    /* Sidebar checkbox text */
    section[data-testid="stSidebar"] .stCheckbox span p {{
        color: {COLORS['text_primary']} !important;
    }}

    /* Sidebar input fields */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [data-baseweb="input"] input,
    section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea {{
        color: {COLORS['text_primary']} !important;
        background-color: {COLORS['card_bg']} !important;
        border-color: {COLORS['gold']}44 !important;
    }}

    /* Sidebar select/dropdown */
    section[data-testid="stSidebar"] [data-baseweb="select"] {{
        color: {COLORS['text_primary']} !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {COLORS['card_bg']} !important;
        border-color: {COLORS['gold']}44 !important;
    }}

    div[data-testid="stMetric"] {{
        background: {COLORS['card_bg']}; border: 1px solid {COLORS['gold']}22;
        border-radius: 12px; padding: 16px 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    div[data-testid="stMetric"] label {{ color: {COLORS['text_muted']} !important; font-size: 0.85rem !important; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: {COLORS['gold']} !important; font-weight: 700 !important; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        background: {COLORS['card_bg']}; border: 1px solid {COLORS['gold']}33;
        border-radius: 8px; color: {COLORS['text_muted']}; padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['gold']}22 !important; border-color: {COLORS['gold']} !important;
        color: {COLORS['gold']} !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['gold']}, #e6c200);
        color: {COLORS['dark_blue']}; font-weight: 700; border: none;
        border-radius: 8px; padding: 0.5rem 2rem; transition: all 0.3s ease;
    }}
    .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(255,215,0,0.3); }}

    .stMultiSelect [data-baseweb="tag"] {{
        background: {COLORS['gold']}22; border: 1px solid {COLORS['gold']}66; color: {COLORS['gold']};
    }}
    hr {{ border-color: {COLORS['gold']}22; }}
</style>
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: UI COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════

def hero_header():
    st.html(f"""
    <div style="background:linear-gradient(135deg,{COLORS['dark_blue']},{COLORS['mid_blue']});
        border:1px solid {COLORS['gold']}33; border-radius:16px; padding:32px 40px;
        margin-bottom:24px; box-shadow:0 8px 32px rgba(0,0,0,0.4); text-align:center; user-select:none;">
        <div style="color:{COLORS['gold']};-webkit-text-fill-color:{COLORS['gold']};
            font-size:2.2rem;font-weight:800;letter-spacing:-0.5px;margin-bottom:8px;">
            📈 Dynamic Efficient Frontier</div>
        <div style="color:{COLORS['light_blue']};-webkit-text-fill-color:{COLORS['light_blue']};
            font-size:1.1rem;font-weight:400;margin-bottom:16px;">
            Multi-Asset Portfolio Optimizer | Modern Portfolio Theory</div>
        <div style="color:{COLORS['gold']}cc;-webkit-text-fill-color:{COLORS['gold']}cc;
            font-size:0.85rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;">
            {PLATFORM}</div>
    </div>""")


def section_header(title: str, subtitle: str = ""):
    sub = f'<div style="color:{COLORS["text_muted"]};-webkit-text-fill-color:{COLORS["text_muted"]};font-size:0.9rem;margin-top:4px;">{subtitle}</div>' if subtitle else ""
    st.html(f"""
    <div style="border-left:4px solid {COLORS['gold']};padding-left:16px;margin:24px 0 16px 0;user-select:none;">
        <div style="color:{COLORS['text_primary']};-webkit-text-fill-color:{COLORS['text_primary']};
            font-size:1.4rem;font-weight:700;">{title}</div>{sub}
    </div>""")


def info_card(label: str, value: str, color: str = COLORS['gold']):
    st.html(f"""
    <div style="background:{COLORS['card_bg']};border:1px solid {color}33;border-radius:12px;
        padding:16px 20px;text-align:center;box-shadow:0 4px 16px rgba(0,0,0,0.2);user-select:none;">
        <div style="color:{COLORS['text_muted']};-webkit-text-fill-color:{COLORS['text_muted']};
            font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{label}</div>
        <div style="color:{color};-webkit-text-fill-color:{color};font-size:1.5rem;font-weight:700;">{value}</div>
    </div>""")


def sidebar_branding():
    st.html(f"""
    <div style="text-align:center;padding:16px 8px;border-bottom:1px solid {COLORS['gold']}22;
        margin-bottom:16px;user-select:none;">
        <div style="color:{COLORS['gold']};-webkit-text-fill-color:{COLORS['gold']};
            font-size:1.1rem;font-weight:700;margin-bottom:4px;">⛰️ The Mountain Path</div>
        <div style="color:{COLORS['text_muted']};-webkit-text-fill-color:{COLORS['text_muted']};
            font-size:0.75rem;">World of Finance</div>
    </div>""")


def app_footer():
    st.html(f"""
    <div style="margin-top:48px;padding:24px;border-top:1px solid {COLORS['gold']}22;text-align:center;user-select:none;">
        <div style="color:{COLORS['text_muted']};-webkit-text-fill-color:{COLORS['text_muted']};
            font-size:0.8rem;margin-bottom:8px;">{AUTHOR} | {AUTHOR_CREDENTIALS}</div>
        <div style="color:{COLORS['text_muted']};-webkit-text-fill-color:{COLORS['text_muted']};
            font-size:0.75rem;margin-bottom:12px;">{PLATFORM}</div>
        <div style="display:flex;justify-content:center;gap:24px;">
            <a href="{LINKEDIN_URL}" target="_blank" style="color:{COLORS['gold']};-webkit-text-fill-color:{COLORS['gold']};
                text-decoration:none;font-size:0.85rem;font-weight:600;">🔗 LinkedIn</a>
            <a href="{GITHUB_URL}" target="_blank" style="color:{COLORS['gold']};-webkit-text-fill-color:{COLORS['gold']};
                text-decoration:none;font-size:0.85rem;font-weight:600;">💻 GitHub</a>
        </div>
    </div>""")


def warning_box(msg: str):
    st.html(f'<div style="background:{COLORS["red"]}15;border:1px solid {COLORS["red"]}44;border-radius:8px;padding:12px 16px;color:{COLORS["red"]};-webkit-text-fill-color:{COLORS["red"]};font-size:0.9rem;user-select:none;">⚠️ {msg}</div>')


def success_box(msg: str):
    st.html(f'<div style="background:{COLORS["green"]}15;border:1px solid {COLORS["green"]}44;border-radius:8px;padding:12px 16px;color:{COLORS["green"]};-webkit-text-fill-color:{COLORS["green"]};font-size:0.9rem;user-select:none;">✅ {msg}</div>')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass
class EfficientFrontierResult:
    sim_returns: np.ndarray
    sim_volatilities: np.ndarray
    sim_sharpe_ratios: np.ndarray
    sim_weights: np.ndarray
    max_sharpe: PortfolioResult
    min_variance: PortfolioResult
    risk_parity: Optional[PortfolioResult] = None
    frontier_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    frontier_volatilities: np.ndarray = field(default_factory=lambda: np.array([]))
    tickers: list = field(default_factory=list)
    individual_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    individual_volatilities: np.ndarray = field(default_factory=lambda: np.array([]))
    cov_matrix: Optional[pd.DataFrame] = None
    corr_matrix: Optional[pd.DataFrame] = None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: PORTFOLIO ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        raise ValueError("No data returned. Check tickers and date range.")
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 valid tickers with overlapping data.")
    return prices


def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=DEFAULT_TRADING_DAYS):
    """Annualized return and volatility for a weight vector."""
    ret = np.sum(mean_returns * weights) * trading_days
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights)))
    return ret, vol


def _neg_sharpe(w, mean_ret, cov, rf, td):
    r, v = portfolio_performance(w, mean_ret, cov, td)
    return -(r - rf) / v


def _port_vol(w, mean_ret, cov, td):
    _, v = portfolio_performance(w, mean_ret, cov, td)
    return v


def _port_ret(w, mean_ret, cov, td):
    r, _ = portfolio_performance(w, mean_ret, cov, td)
    return r


def _risk_parity_obj(w, cov, td):
    w = np.array(w)
    cov_ann = cov * td
    pv = np.sqrt(w.T @ cov_ann @ w)
    mc = cov_ann @ w
    rc = w * mc / pv
    target = pv / len(w)
    return np.sum((rc - target) ** 2)


def optimize_max_sharpe(mean_ret, cov, rf=DEFAULT_RISK_FREE_RATE, td=DEFAULT_TRADING_DAYS, short=False):
    n = len(mean_ret)
    bounds = ((-1, 1) if short else (0, 1),) * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(_neg_sharpe, np.ones(n)/n, args=(mean_ret, cov, rf, td), method="SLSQP", bounds=bounds, constraints=cons)
    r, v = portfolio_performance(res.x, mean_ret, cov, td)
    return PortfolioResult(weights=res.x, expected_return=r, volatility=v, sharpe_ratio=(r-rf)/v)


def optimize_min_variance(mean_ret, cov, rf=DEFAULT_RISK_FREE_RATE, td=DEFAULT_TRADING_DAYS, short=False):
    n = len(mean_ret)
    bounds = ((-1, 1) if short else (0, 1),) * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(_port_vol, np.ones(n)/n, args=(mean_ret, cov, td), method="SLSQP", bounds=bounds, constraints=cons)
    r, v = portfolio_performance(res.x, mean_ret, cov, td)
    return PortfolioResult(weights=res.x, expected_return=r, volatility=v, sharpe_ratio=(r-rf)/v)


def optimize_risk_parity(mean_ret, cov, rf=DEFAULT_RISK_FREE_RATE, td=DEFAULT_TRADING_DAYS):
    n = len(mean_ret)
    bounds = ((0.01, 1),) * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(_risk_parity_obj, np.ones(n)/n, args=(cov, td), method="SLSQP", bounds=bounds, constraints=cons)
    r, v = portfolio_performance(res.x, mean_ret, cov, td)
    return PortfolioResult(weights=res.x, expected_return=r, volatility=v, sharpe_ratio=(r-rf)/v)


def compute_frontier_curve(mean_ret, cov, rf=DEFAULT_RISK_FREE_RATE, td=DEFAULT_TRADING_DAYS, n_pts=100, short=False):
    """Trace efficient frontier by minimizing vol at each target return."""
    n = len(mean_ret)
    bounds = ((-1, 1) if short else (0, 1),) * n
    init = np.ones(n) / n
    min_var = optimize_min_variance(mean_ret, cov, rf, td, short)
    max_r = float(np.max(mean_ret) * td)
    targets = np.linspace(min_var.expected_return, max_r, n_pts)
    f_vols, f_rets = [], []
    for t in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tgt=t: _port_ret(w, mean_ret, cov, td) - tgt},
        ]
        res = minimize(_port_vol, init, args=(mean_ret, cov, td), method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            _, v = portfolio_performance(res.x, mean_ret, cov, td)
            f_vols.append(v)
            f_rets.append(t)
    return np.array(f_vols), np.array(f_rets)


def run_monte_carlo(mean_ret, cov, n_port=10000, rf=DEFAULT_RISK_FREE_RATE, td=DEFAULT_TRADING_DAYS):
    """Generate random portfolios via Dirichlet-distributed weights."""
    n = len(mean_ret)
    rets = np.zeros(n_port)
    vols = np.zeros(n_port)
    sharps = np.zeros(n_port)
    wts = np.zeros((n_port, n))
    for i in range(n_port):
        w = np.random.dirichlet(np.ones(n))
        r, v = portfolio_performance(w, mean_ret, cov, td)
        rets[i], vols[i], sharps[i], wts[i] = r, v, (r - rf) / v, w
    return rets, vols, sharps, wts


def run_efficient_frontier(prices, n_port=10000, rf=DEFAULT_RISK_FREE_RATE, td=DEFAULT_TRADING_DAYS, short=False):
    """Full pipeline: returns → Monte Carlo → optimization → frontier curve."""
    daily_ret = prices.pct_change().dropna()
    mean_ret = daily_ret.mean().values
    cov = daily_ret.cov().values
    tickers = list(prices.columns)

    sim_r, sim_v, sim_s, sim_w = run_monte_carlo(mean_ret, cov, n_port, rf, td)
    ms = optimize_max_sharpe(mean_ret, cov, rf, td, short)
    mv = optimize_min_variance(mean_ret, cov, rf, td, short)
    rp = optimize_risk_parity(mean_ret, cov, rf, td)
    f_vols, f_rets = compute_frontier_curve(mean_ret, cov, rf, td, 100, short)

    return EfficientFrontierResult(
        sim_returns=sim_r, sim_volatilities=sim_v, sim_sharpe_ratios=sim_s, sim_weights=sim_w,
        max_sharpe=ms, min_variance=mv, risk_parity=rp,
        frontier_returns=f_rets, frontier_volatilities=f_vols,
        tickers=tickers,
        individual_returns=mean_ret * td,
        individual_volatilities=daily_ret.std().values * np.sqrt(td),
        cov_matrix=pd.DataFrame(cov * td, index=tickers, columns=tickers),
        corr_matrix=daily_ret.corr(),
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: PLOTLY CHART TEMPLATE
# ═════════════════════════════════════════════════════════════════════════════

def _styled(fig):
    """Apply Mountain Path theme to a Plotly figure (compatible with all Plotly versions)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,25,47,0.6)",
        font=dict(color=COLORS["text_primary"], family="Segoe UI, sans-serif"),
        title=dict(font=dict(color=COLORS["gold"], size=18)),
        legend=dict(bgcolor="rgba(17,34,64,0.8)", bordercolor="rgba(255,215,0,0.2)", borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor=COLORS["card_bg"], bordercolor=COLORS["gold"], font=dict(color=COLORS["text_primary"])),
    )
    fig.update_xaxes(gridcolor="rgba(255,215,0,0.08)", zerolinecolor="rgba(255,215,0,0.15)")
    fig.update_yaxes(gridcolor="rgba(255,215,0,0.08)", zerolinecolor="rgba(255,215,0,0.15)")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: CHART BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def plot_efficient_frontier(res, rf, show_cml=True, show_ind=True):
    fig = go.Figure()

    # Monte Carlo cloud
    fig.add_trace(go.Scatter(
        x=res.sim_volatilities*100, y=res.sim_returns*100, mode="markers",
        marker=dict(size=3, color=res.sim_sharpe_ratios, opacity=0.5,
            colorscale=[[0,COLORS["red"]],[0.5,COLORS["light_blue"]],[1,COLORS["gold"]]],
            colorbar=dict(title=dict(text="Sharpe<br>Ratio",font=dict(size=11)),thickness=15,len=0.6)),
        text=[f"Ret:{r:.2f}% Vol:{v:.2f}% SR:{s:.3f}" for r,v,s in
              zip(res.sim_returns*100, res.sim_volatilities*100, res.sim_sharpe_ratios)],
        hoverinfo="text", name="Simulated Portfolios"))

    # Efficient frontier curve
    if len(res.frontier_volatilities) > 0:
        fig.add_trace(go.Scatter(
            x=res.frontier_volatilities*100, y=res.frontier_returns*100,
            mode="lines", line=dict(color=COLORS["gold"], width=3), name="Efficient Frontier"))

    # Max Sharpe
    fig.add_trace(go.Scatter(
        x=[res.max_sharpe.volatility*100], y=[res.max_sharpe.expected_return*100],
        mode="markers", marker=dict(size=18, color=COLORS["gold"], symbol="star", line=dict(color="white",width=2)),
        name=f"Max Sharpe ({res.max_sharpe.sharpe_ratio:.3f})",
        hovertext=f"Ret:{res.max_sharpe.expected_return*100:.2f}% Vol:{res.max_sharpe.volatility*100:.2f}%", hoverinfo="text"))

    # Min Variance
    fig.add_trace(go.Scatter(
        x=[res.min_variance.volatility*100], y=[res.min_variance.expected_return*100],
        mode="markers", marker=dict(size=16, color=COLORS["cyan"], symbol="diamond", line=dict(color="white",width=2)),
        name=f"Min Variance (σ={res.min_variance.volatility*100:.2f}%)",
        hovertext=f"Ret:{res.min_variance.expected_return*100:.2f}% Vol:{res.min_variance.volatility*100:.2f}%", hoverinfo="text"))

    # Risk Parity
    if res.risk_parity:
        fig.add_trace(go.Scatter(
            x=[res.risk_parity.volatility*100], y=[res.risk_parity.expected_return*100],
            mode="markers", marker=dict(size=16, color=COLORS["green"], symbol="hexagon", line=dict(color="white",width=2)),
            name="Risk Parity",
            hovertext=f"Ret:{res.risk_parity.expected_return*100:.2f}% Vol:{res.risk_parity.volatility*100:.2f}%", hoverinfo="text"))

    # CML
    if show_cml:
        max_vol = np.max(res.sim_volatilities)*100*1.1
        fig.add_trace(go.Scatter(
            x=[0, max_vol], y=[rf*100, rf*100 + res.max_sharpe.sharpe_ratio*max_vol],
            mode="lines", line=dict(color=COLORS["gold"], width=2, dash="dash"), name="Capital Market Line"))

    # Individual assets
    if show_ind:
        for i, t in enumerate(res.tickers):
            fig.add_trace(go.Scatter(
                x=[res.individual_volatilities[i]*100], y=[res.individual_returns[i]*100],
                mode="markers+text", marker=dict(size=12, color=PLOTLY_COLORS[i%len(PLOTLY_COLORS)],
                    symbol="circle", line=dict(color="white",width=1.5)),
                text=[t], textposition="top center", textfont=dict(size=10, color=COLORS["text_primary"]),
                name=t, showlegend=False))

    fig.update_layout(title="Efficient Frontier — Mean-Variance Optimization",
        xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
        height=650, margin=dict(l=60,r=40,t=60,b=60))
    return _styled(fig)


def plot_weight_comparison(res):
    fig = go.Figure()
    w = 0.25
    fig.add_trace(go.Bar(x=res.tickers, y=res.max_sharpe.weights*100, name="Max Sharpe", marker_color=COLORS["gold"], width=w, offset=-w))
    fig.add_trace(go.Bar(x=res.tickers, y=res.min_variance.weights*100, name="Min Variance", marker_color=COLORS["cyan"], width=w, offset=0))
    if res.risk_parity:
        fig.add_trace(go.Bar(x=res.tickers, y=res.risk_parity.weights*100, name="Risk Parity", marker_color=COLORS["green"], width=w, offset=w))
    fig.update_layout(title="Portfolio Weight Allocation Comparison", xaxis_title="Assets", yaxis_title="Weight (%)", barmode="group", height=450)
    return _styled(fig)


def plot_weight_pies(res):
    fig = make_subplots(rows=1, cols=3, specs=[[{"type":"pie"},{"type":"pie"},{"type":"pie"}]],
                        subplot_titles=["Max Sharpe","Min Variance","Risk Parity"])
    for port, col in [(res.max_sharpe,1),(res.min_variance,2),(res.risk_parity,3)]:
        if port is None: continue
        mask = port.weights > 0.005
        fig.add_trace(go.Pie(
            labels=[t for t,m in zip(res.tickers,mask) if m],
            values=[w*100 for w,m in zip(port.weights,mask) if m],
            hole=0.45, marker=dict(colors=PLOTLY_COLORS[:sum(mask)]),
            textinfo="label+percent", textfont=dict(size=10)), row=1, col=col)
    fig.update_layout(title="Weight Allocation — Donut Charts", height=400, showlegend=False)
    return _styled(fig)


def plot_correlation_matrix(res):
    corr = res.corr_matrix
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,COLORS["red"]],[0.5,COLORS["card_bg"]],[1,COLORS["gold"]]],
        zmin=-1, zmax=1, text=np.round(corr.values,3), texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(title="ρ",thickness=15)))
    fig.update_layout(title="Asset Return Correlation Matrix", height=500)
    return _styled(fig)


def plot_covariance_matrix(res):
    cov = res.cov_matrix
    fig = go.Figure(data=go.Heatmap(
        z=cov.values, x=cov.columns.tolist(), y=cov.index.tolist(),
        colorscale=[[0,COLORS["mid_blue"]],[1,COLORS["gold"]]],
        text=np.round(cov.values,6), texttemplate="%{text:.5f}",
        textfont=dict(size=10), colorbar=dict(title="Cov",thickness=15)))
    fig.update_layout(title="Annualized Covariance Matrix", height=500)
    return _styled(fig)


def plot_risk_return_bars(res):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=res.tickers, y=res.individual_returns*100, name="Ann. Return (%)", marker_color=COLORS["gold"]))
    fig.add_trace(go.Bar(x=res.tickers, y=res.individual_volatilities*100, name="Ann. Volatility (%)", marker_color=COLORS["cyan"]))
    fig.update_layout(title="Individual Asset Risk vs Return", barmode="group", xaxis_title="Assets", yaxis_title="(%)", height=400)
    return _styled(fig)


def plot_price_history(prices):
    norm = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col,
            line=dict(color=PLOTLY_COLORS[i%len(PLOTLY_COLORS)], width=2)))
    fig.update_layout(title="Normalized Price History (Base = 100)", xaxis_title="Date", yaxis_title="Indexed Price", height=450, hovermode="x unified")
    return _styled(fig)


def plot_drawdowns(prices):
    rets = prices.pct_change().dropna()
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax() * 100
    fig = go.Figure()
    for i, col in enumerate(dd.columns):
        c = PLOTLY_COLORS[i%len(PLOTLY_COLORS)]
        rgb = ",".join(str(int(c.lstrip("#")[j:j+2],16)) for j in (0,2,4))
        fig.add_trace(go.Scatter(x=dd.index, y=dd[col], name=col,
            line=dict(color=c, width=1.5), fill="tozeroy", fillcolor=f"rgba({rgb},0.1)"))
    fig.update_layout(title="Asset Drawdowns (%)", xaxis_title="Date", yaxis_title="Drawdown (%)", height=400, hovermode="x unified")
    return _styled(fig)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
st.html(PAGE_CSS)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    sidebar_branding()
    st.markdown("### 🎯 Portfolio Configuration")

    ticker_input = st.text_area("Enter Tickers (one per line)", value="\n".join(DEFAULT_TICKERS), height=150,
        help="Yahoo Finance tickers. Use .NS for NSE, .BO for BSE.")
    tickers = [t.strip().upper() for t in ticker_input.strip().split("\n") if t.strip()]

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=datetime.now()-timedelta(days=3*365), max_value=datetime.now()-timedelta(days=30))
    with c2:
        end_date = st.date_input("End Date", value=datetime.now(), max_value=datetime.now())

    st.markdown("---")
    st.markdown("### ⚙️ Simulation Parameters")

    n_portfolios = st.slider("Monte Carlo Simulations", 1000, 50000, DEFAULT_NUM_PORTFOLIOS, 1000,
        help="Higher = more accurate, slower.")
    risk_free_rate = st.number_input("Risk-Free Rate (annual %)", 0.0, 20.0, DEFAULT_RISK_FREE_RATE*100, 0.25,
        help="India 10Y ≈ 7%. US T-Bill ≈ 5%.") / 100
    trading_days = st.number_input("Trading Days / Year", 200, 365, DEFAULT_TRADING_DAYS, 1)
    allow_short = st.checkbox("Allow Short Selling", False)

    st.markdown("---")
    st.markdown("### 📊 Display Options")
    show_cml = st.checkbox("Show Capital Market Line", True)
    show_individual = st.checkbox("Show Individual Assets", True)

    st.markdown("---")
    run_button = st.button("🚀 Run Optimization", use_container_width=True, type="primary")


# ── Main Content ──────────────────────────────────────────────────────────────

hero_header()

if run_button or "result" in st.session_state:

    if run_button:
        if len(tickers) < 2:
            warning_box("Please enter at least 2 tickers.")
            st.stop()
        if start_date >= end_date:
            warning_box("Start date must be before end date.")
            st.stop()

        with st.spinner("📡 Fetching data from Yahoo Finance..."):
            try:
                prices = fetch_stock_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            except Exception as e:
                warning_box(f"Data fetch error: {e}")
                st.stop()

        with st.spinner(f"⚡ Running {n_portfolios:,} Monte Carlo simulations & optimization..."):
            try:
                result = run_efficient_frontier(prices, n_portfolios, risk_free_rate, trading_days, allow_short)
            except Exception as e:
                warning_box(f"Optimization error: {e}")
                st.stop()

        st.session_state["result"] = result
        st.session_state["prices"] = prices
        st.session_state["rf"] = risk_free_rate
        st.session_state["show_cml"] = show_cml
        st.session_state["show_ind"] = show_individual

        success_box(f"Analysis complete — {len(result.tickers)} assets ({prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')})")

    result = st.session_state["result"]
    prices = st.session_state["prices"]
    rf = st.session_state.get("rf", risk_free_rate)
    s_cml = st.session_state.get("show_cml", show_cml)
    s_ind = st.session_state.get("show_ind", show_individual)

    # ── Key Metrics ───────────────────────────────────────────────────────
    section_header("Key Portfolio Metrics", "Optimized portfolio summary")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: info_card("Max Sharpe Ratio", f"{result.max_sharpe.sharpe_ratio:.3f}", COLORS["gold"])
    with c2: info_card("Max Sharpe Return", f"{result.max_sharpe.expected_return*100:.2f}%", COLORS["green"])
    with c3: info_card("Max Sharpe Risk", f"{result.max_sharpe.volatility*100:.2f}%", COLORS["red"])
    with c4: info_card("Min Var Return", f"{result.min_variance.expected_return*100:.2f}%", COLORS["cyan"])
    with c5: info_card("Min Var Risk", f"{result.min_variance.volatility*100:.2f}%", COLORS["cyan"])
    with c6: info_card("Assets Analyzed", f"{len(result.tickers)}", COLORS["light_blue"])

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Efficient Frontier", "⚖️ Weight Allocation",
        "🔗 Correlation & Covariance", "📊 Asset Analysis", "📋 Detailed Results"])

    with tab1:
        section_header("Efficient Frontier", "Monte Carlo simulation with optimized portfolios")
        st.plotly_chart(plot_efficient_frontier(result, rf, s_cml, s_ind), use_container_width=True)
        with st.expander("📘 Understanding the Efficient Frontier"):
            st.markdown("""
            - **⭐ Gold Star** — Maximum Sharpe Ratio (Tangency) Portfolio
            - **💎 Cyan Diamond** — Global Minimum Variance Portfolio
            - **🟢 Green Hexagon** — Risk Parity Portfolio
            - **Dashed Gold Line** — Capital Market Line
            - **Color Cloud** — Monte Carlo simulated portfolios (colored by Sharpe)
            """)

    with tab2:
        section_header("Portfolio Weight Allocation", "Comparing optimal allocations")
        st.plotly_chart(plot_weight_comparison(result), use_container_width=True)
        st.plotly_chart(plot_weight_pies(result), use_container_width=True)
        section_header("Weight Details")
        wdf = pd.DataFrame({
            "Asset": result.tickers,
            "Max Sharpe (%)": np.round(result.max_sharpe.weights*100, 2),
            "Min Variance (%)": np.round(result.min_variance.weights*100, 2),
            "Risk Parity (%)": np.round(result.risk_parity.weights*100, 2) if result.risk_parity else 0,
        })
        st.dataframe(wdf, use_container_width=True, hide_index=True)

    with tab3:
        ca, cb = st.columns(2)
        with ca:
            section_header("Correlation Matrix")
            st.plotly_chart(plot_correlation_matrix(result), use_container_width=True)
        with cb:
            section_header("Covariance Matrix")
            st.plotly_chart(plot_covariance_matrix(result), use_container_width=True)
        with st.expander("📘 Interpreting Correlation & Covariance"):
            st.markdown("""
            - **Correlation (ρ)**: -1 to +1. Low/negative = diversification benefit.
            - **Covariance**: Co-movement magnitude. Portfolio risk = w'Σw.
            """)

    with tab4:
        section_header("Individual Asset Performance")
        st.plotly_chart(plot_risk_return_bars(result), use_container_width=True)
        section_header("Normalized Price History", "Rebased to 100")
        st.plotly_chart(plot_price_history(prices), use_container_width=True)
        section_header("Drawdown Analysis", "Peak-to-trough declines")
        st.plotly_chart(plot_drawdowns(prices), use_container_width=True)

    with tab5:
        section_header("Optimized Portfolio Summary")
        portfolios = {"Max Sharpe": result.max_sharpe, "Min Variance": result.min_variance}
        if result.risk_parity: portfolios["Risk Parity"] = result.risk_parity
        summary = [{"Portfolio": n, "Return (%)": round(p.expected_return*100,2),
                     "Volatility (%)": round(p.volatility*100,2), "Sharpe": round(p.sharpe_ratio,4)}
                    for n,p in portfolios.items()]
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        section_header("Individual Asset Statistics")
        adf = pd.DataFrame({
            "Asset": result.tickers,
            "Ann. Return (%)": np.round(result.individual_returns*100, 2),
            "Ann. Volatility (%)": np.round(result.individual_volatilities*100, 2),
            "Return/Risk": np.round(result.individual_returns/result.individual_volatilities, 3),
        })
        st.dataframe(adf, use_container_width=True, hide_index=True)

        section_header("Export Results")
        @st.cache_data
        def to_csv(df): return df.to_csv(index=False).encode("utf-8")
        ce1, ce2 = st.columns(2)
        with ce1: st.download_button("📥 Download Weights (CSV)", to_csv(wdf), "portfolio_weights.csv", "text/csv")
        with ce2: st.download_button("📥 Download Asset Stats (CSV)", to_csv(adf), "asset_statistics.csv", "text/csv")

        section_header("Simulation Statistics")
        st.dataframe(pd.DataFrame({
            "Metric": ["Portfolios Simulated","Best Sharpe (MC)","Worst Sharpe (MC)","Mean Return (MC)","Mean Volatility (MC)"],
            "Value": [f"{len(result.sim_returns):,}", f"{result.sim_sharpe_ratios.max():.4f}",
                      f"{result.sim_sharpe_ratios.min():.4f}", f"{result.sim_returns.mean()*100:.2f}%",
                      f"{result.sim_volatilities.mean()*100:.2f}%"],
        }), use_container_width=True, hide_index=True)

else:
    section_header("Getting Started", "Configure your portfolio in the sidebar and click Run")
    st.markdown("""
    This application implements **Modern Portfolio Theory (Markowitz, 1952)** to find
    optimal multi-asset allocations using real market data from Yahoo Finance.

    **Methodology:** Fetch prices → Compute returns & covariance → Monte Carlo simulation (Dirichlet weights)
    → Constrained optimization (Max Sharpe, Min Variance, Risk Parity) → Trace efficient frontier via SQP
    → Overlay Capital Market Line.

    **Defaults:** 5 major NSE stocks, 3-year lookback, 10,000 simulations, Rf = 7%.
    """)
    with st.expander("📘 Ticker Formats"):
        st.markdown("""
        | Market | Suffix | Example |
        |--------|--------|---------|
        | NSE India | `.NS` | `RELIANCE.NS` |
        | BSE India | `.BO` | `TCS.BO` |
        | US | None | `AAPL`, `MSFT` |
        | London | `.L` | `HSBA.L` |
        | Hong Kong | `.HK` | `0700.HK` |
        """)

app_footer()
