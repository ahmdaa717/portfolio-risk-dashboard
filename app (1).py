import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0d1117; color: #e6edf3; }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7d8590;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 500;
        color: #e6edf3;
    }
    .metric-value.negative { color: #f85149; }
    .metric-value.positive { color: #3fb950; }
    .metric-value.neutral  { color: #d29922; }

    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #7d8590;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    .stress-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
    .stress-table th { background: #21262d; color: #7d8590; padding: 10px 14px; text-align: right; font-weight: 500; letter-spacing: 0.05em; }
    .stress-table th:first-child { text-align: left; }
    .stress-table td { padding: 10px 14px; border-bottom: 1px solid #21262d; text-align: right; }
    .stress-table td:first-child { text-align: left; color: #7d8590; }
    .stress-table tr:last-child td { border-bottom: none; }
    .red { color: #f85149; } .green { color: #3fb950; } .yellow { color: #d29922; }

    div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── DATA & COMPUTATION ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    returns = np.log(df / df.shift(1)).dropna()
    return df, returns


def compute_metrics(port_returns, confidence=0.95):
    mu = port_returns.mean()
    sigma = port_returns.std()
    rf = 0.02 / 252

    var_hist  = np.percentile(port_returns, (1 - confidence) * 100)
    var_param = mu - 1.645 * sigma
    np.random.seed(42)
    sim = np.random.normal(mu, sigma, 10000)
    var_mc    = np.percentile(sim, (1 - confidence) * 100)
    cvar      = port_returns[port_returns <= var_hist].mean()

    sharpe  = (mu - rf) / sigma * np.sqrt(252)
    ds      = port_returns[port_returns < 0].std()
    sortino = (mu - rf) / ds * np.sqrt(252)

    eq = (1 + port_returns).cumprod()
    rm = eq.cummax()
    dd = (eq - rm) / rm
    max_dd = dd.min()

    return {
        "var_hist": var_hist, "var_param": var_param, "var_mc": var_mc,
        "cvar": cvar, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "equity": eq, "drawdown": dd,
        "ann_return": mu * 252, "ann_vol": sigma * np.sqrt(252)
    }


def stress_metrics(crisis_returns, confidence=0.95):
    var   = np.percentile(crisis_returns, (1 - confidence) * 100)
    cvar  = crisis_returns[crisis_returns <= var].mean()
    eq    = (1 + crisis_returns).cumprod()
    rm    = eq.cummax()
    dd    = (eq - rm) / rm
    return {"var": var, "cvar": cvar, "max_dd": dd.min()}


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Portfolio Settings")
    st.markdown("---")

    default_tickers = "SPY, QQQ, GLD, TLT, EEM, BTC-USD"
    ticker_input = st.text_input("Tickers (comma-separated)", value=default_tickers)
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.to_datetime("2024-01-01"))

    st.markdown("**Portfolio Weights**")
    weights = []
    cols = st.columns(2)
    for i, ticker in enumerate(tickers):
        w = cols[i % 2].number_input(ticker, min_value=0.0, max_value=1.0,
                                      value=round(1/len(tickers), 4), step=0.01, key=f"w_{i}")
        weights.append(w)

    weights = np.array(weights)
    w_sum = weights.sum()
    if abs(w_sum - 1.0) > 0.001:
        st.warning(f"Weights sum to {w_sum:.3f} — must equal 1.0")
        weights = weights / w_sum

    confidence = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
    st.markdown("---")
    run = st.button("▶  Run Analysis", use_container_width=True, type="primary")


# ─── MAIN ───────────────────────────────────────────────────────────────────
st.markdown("## Portfolio Risk Dashboard")
st.markdown("<p style='color:#7d8590;margin-top:-12px;'>Multi-Asset Risk Engine · VaR · CVaR · Stress Testing</p>", unsafe_allow_html=True)

if not run:
    st.info("Configure your portfolio in the sidebar and click **Run Analysis**.")
    st.stop()

with st.spinner("Fetching market data..."):
    try:
        prices, returns = load_data(tickers, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

# filter weights to available tickers
available = [t for t in tickers if t in returns.columns]
weights_map = {t: w for t, w in zip(tickers, weights)}
w = np.array([weights_map[t] for t in available])
w = w / w.sum()

port_returns = returns[available] @ w
m = compute_metrics(port_returns, confidence)

# ─── KPI ROW ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Risk Metrics</div>', unsafe_allow_html=True)

def card(label, value, fmt=".2%", color="neutral"):
    formatted = f"{value:{fmt}}" if fmt else value
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{formatted}</div>
    </div>"""

kpi_cols = st.columns(7)
kpis = [
    ("Ann. Return",  m["ann_return"],  ".2%", "positive" if m["ann_return"] > 0 else "negative"),
    ("Ann. Vol",     m["ann_vol"],     ".2%", "neutral"),
    ("Hist VaR",     m["var_hist"],    ".2%", "negative"),
    ("CVaR",         m["cvar"],        ".2%", "negative"),
    ("Sharpe",       m["sharpe"],      ".2f", "positive" if m["sharpe"] > 0 else "negative"),
    ("Sortino",      m["sortino"],     ".2f", "positive" if m["sortino"] > 0 else "negative"),
    ("Max Drawdown", m["max_dd"],      ".2%", "negative"),
]
for col, (label, value, fmt, color) in zip(kpi_cols, kpis):
    col.markdown(card(label, value, fmt, color), unsafe_allow_html=True)


# ─── EQUITY CURVE + DRAWDOWN ────────────────────────────────────────────────
st.markdown('<div class="section-header">Equity Curve & Drawdown</div>', unsafe_allow_html=True)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.04)

fig.add_trace(go.Scatter(
    x=m["equity"].index, y=m["equity"].values,
    name="Portfolio", line=dict(color="#3fb950", width=1.8),
    fill="tozeroy", fillcolor="rgba(63,185,80,0.06)"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=m["drawdown"].index, y=m["drawdown"].values,
    name="Drawdown", line=dict(color="#f85149", width=1.2),
    fill="tozeroy", fillcolor="rgba(248,81,73,0.12)"
), row=2, col=1)

fig.update_layout(
    height=480, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(family="Inter", color="#7d8590", size=11),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
    margin=dict(l=0, r=0, t=8, b=0),
    hovermode="x unified"
)
fig.update_xaxes(gridcolor="#21262d", showgrid=True, zeroline=False)
fig.update_yaxes(gridcolor="#21262d", showgrid=True, zeroline=False)
st.plotly_chart(fig, use_container_width=True)


# ─── VAR COMPARISON + RETURN DIST ───────────────────────────────────────────
st.markdown('<div class="section-header">VaR Comparison & Return Distribution</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    fig_var = go.Figure()
    methods = ["Historical", "Parametric", "Monte Carlo"]
    values  = [m["var_hist"], m["var_param"], m["var_mc"]]
    colors  = ["#f85149", "#d29922", "#388bfd"]
    fig_var.add_trace(go.Bar(
        x=methods, y=values, marker_color=colors,
        text=[f"{v:.2%}" for v in values], textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#e6edf3")
    ))
    fig_var.update_layout(
        title=dict(text=f"VaR at {confidence:.0%} Confidence", font=dict(size=13, color="#7d8590")),
        height=320, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(family="Inter", color="#7d8590"),
        margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        yaxis=dict(gridcolor="#21262d", tickformat=".1%")
    )
    st.plotly_chart(fig_var, use_container_width=True)

with col_r:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=port_returns.values, nbinsx=80,
        marker_color="#388bfd", opacity=0.7, name="Returns"
    ))
    fig_dist.add_vline(x=m["var_hist"],  line_color="#f85149", line_dash="dash",
                       annotation_text="VaR", annotation_font_color="#f85149")
    fig_dist.add_vline(x=m["cvar"],      line_color="#d29922", line_dash="dot",
                       annotation_text="CVaR", annotation_font_color="#d29922")
    fig_dist.update_layout(
        title=dict(text="Daily Return Distribution", font=dict(size=13, color="#7d8590")),
        height=320, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(family="Inter", color="#7d8590"),
        margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d")
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ─── CORRELATION HEATMAP ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Asset Correlation Matrix</div>', unsafe_allow_html=True)

corr = returns[available].corr()
fig_corr = go.Figure(go.Heatmap(
    z=corr.values, x=available, y=available,
    colorscale=[[0, "#f85149"], [0.5, "#161b22"], [1, "#3fb950"]],
    zmin=-1, zmax=1, text=np.round(corr.values, 2),
    texttemplate="%{text}", textfont=dict(family="JetBrains Mono", size=12),
    hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>"
))
fig_corr.update_layout(
    height=380, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Inter", color="#7d8590"),
    margin=dict(l=0, r=0, t=8, b=0)
)
st.plotly_chart(fig_corr, use_container_width=True)


# ─── STRESS TESTING ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Stress Test — Crisis Scenarios</div>', unsafe_allow_html=True)

scenarios = {
    "Normal Period":   (str(start_date), str(end_date)),
    "COVID Crash":     ("2020-02-01", "2020-04-30"),
    "2022 Rate Hike":  ("2022-01-01", "2022-12-31"),
}

results = {}
for name, (s, e) in scenarios.items():
    sl = port_returns.loc[s:e]
    if len(sl) > 10:
        results[name] = stress_metrics(sl, confidence)

def fmt_cell(v):
    color = "red" if v < -0.03 else "yellow" if v < 0 else "green"
    return f'<span class="{color}">{v:.2%}</span>'

rows = ""
for name, r in results.items():
    rows += f"""
    <tr>
        <td>{name}</td>
        <td>{fmt_cell(r['var'])}</td>
        <td>{fmt_cell(r['cvar'])}</td>
        <td>{fmt_cell(r['max_dd'])}</td>
    </tr>"""

st.markdown(f"""
<table class="stress-table">
  <thead>
    <tr>
      <th>Scenario</th>
      <th>VaR ({confidence:.0%})</th>
      <th>CVaR ({confidence:.0%})</th>
      <th>Max Drawdown</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)


# ─── INDIVIDUAL ASSET METRICS ───────────────────────────────────────────────
st.markdown('<div class="section-header">Individual Asset Risk</div>', unsafe_allow_html=True)

asset_rows = ""
for ticker in available:
    r = returns[ticker]
    v = np.percentile(r, 5)
    s = (r.mean() - 0.02/252) / r.std() * np.sqrt(252)
    asset_rows += f"""
    <tr>
        <td style="color:#e6edf3;font-weight:500">{ticker}</td>
        <td>{fmt_cell(v)}</td>
        <td>{fmt_cell(r[r <= v].mean())}</td>
        <td style="color:#{'3fb950' if s>0.5 else 'd29922' if s>0 else 'f85149'}">{s:.2f}</td>
        <td style="color:#7d8590;font-family:'JetBrains Mono',monospace">{r.std()*np.sqrt(252):.2%}</td>
    </tr>"""

st.markdown(f"""
<table class="stress-table">
  <thead>
    <tr><th>Asset</th><th>VaR (95%)</th><th>CVaR (95%)</th><th>Sharpe</th><th>Ann. Vol</th></tr>
  </thead>
  <tbody>{asset_rows}</tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<br><p style='color:#21262d;text-align:center;font-size:11px'>Portfolio Risk Dashboard · Built with Python & Streamlit</p>", unsafe_allow_html=True)
