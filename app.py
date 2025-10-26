

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RiskLens — Portfolio Risk & Scenario Simulator", layout="wide")

# ---------- Helper functions & caching ----------
@st.cache_data(ttl=60*30)
def fetch_adjusted_close(tickers, period="1y", interval="1d"):
    """
    Fetch adjusted close prices for tickers using yfinance.
    Returns DataFrame with columns = tickers, index = datetime
    """
    # yfinance can accept list
    try:
        df = yf.download(tickers, period=period, interval=interval, group_by='ticker', auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        st.error(f"yfinance download error: {e}")
        return pd.DataFrame()

    # Normalize structure
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex: top level tickers
        out = {}
        for tk in tickers:
            try:
                ser = df[tk]["Adj Close"].rename(tk)
                out[tk] = ser
            except Exception:
                # fallback if structure different
                try:
                    ser = df["Adj Close"][tk].rename(tk)
                    out[tk] = ser
                except Exception:
                    out[tk] = pd.Series(dtype=float)
        prices = pd.DataFrame(out)
    else:
        # Single ticker case or single-level columns
        if "Adj Close" in df.columns:
            prices = pd.DataFrame({tickers[0]: df["Adj Close"]}) if len(tickers) == 1 else df["Adj Close"]
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
        else:
            prices = df
    # drop completely empty columns
    prices = prices.dropna(axis=1, how='all')
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices

@st.cache_data(ttl=60*30)
def fetch_latest_price(ticker):
    """Return latest close price for a single ticker using yfinance Ticker.history."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d", auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

def compute_simple_returns(price_ser):
    """Simple returns percentage series (not log)."""
    return price_ser.pct_change().dropna()

def annualize_volatility(daily_returns, trading_days=252):
    return daily_returns.std() * np.sqrt(trading_days)

def compute_factor_betas(stock_ret, factors_df):
    """
    Regress stock_ret on factors_df (DataFrame of factor returns).
    Returns (beta_series, alpha, r_squared)
    Uses OLS (statsmodels).
    """
    df = pd.concat([stock_ret, factors_df], axis=1).dropna()
    if df.shape[0] < 30:
        return None, None, None  # not enough data
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    params = model.params
    # first param is const (alpha), rest are betas with factor names
    alpha = params.get("const", np.nan)
    betas = params.drop("const", errors='ignore')
    r2 = model.rsquared
    return betas, alpha, r2

def portfolio_metrics(df_holdings, prices_df, bench_symbol):
    """
    Given holdings DataFrame with columns: ticker, shares, avg_price
    and prices_df containing adjusted close price history for tickers and bench,
    compute latest prices, market values, weights, returns, betas etc.
    Returns enriched holdings df and intermediate data.
    """
    holdings = df_holdings.copy()
    tickers = holdings['ticker'].tolist()

    # Latest prices: try prices_df last available, otherwise yfinance single fetch
    latest_prices = {}
    for tk in tickers:
        if tk in prices_df.columns and not prices_df[tk].dropna().empty:
            latest_prices[tk] = float(prices_df[tk].dropna().iloc[-1])
        else:
            latest_prices[tk] = fetch_latest_price(tk)

    holdings['market_price'] = holdings['ticker'].map(latest_prices)
    holdings['market_value'] = holdings['market_price'] * holdings['shares']
    holdings['cost_basis'] = holdings['avg_price'] * holdings['shares']
    holdings['unrealized_pnl'] = holdings['market_value'] - holdings['cost_basis']
    total_value = holdings['market_value'].sum()
    holdings['weight'] = holdings['market_value'] / total_value

    # Returns
    # Build returns df for ticks we have
    rets = prices_df[tickers].pct_change().dropna(how='all')
    bench_ret = prices_df[bench_symbol].pct_change().dropna() if bench_symbol in prices_df.columns else pd.Series(dtype=float)

    # Per-stock volatility and beta vs multi-factor
    holdings_metrics = []
    # prepare factors DF: market, oil, fx
    factor_cols = [c for c in prices_df.columns if c not in tickers]
    factors_df = prices_df[factor_cols].pct_change().dropna() if len(factor_cols) > 0 else pd.DataFrame()

    for idx, row in holdings.iterrows():
        tk = row['ticker']
        # volatility
        vol = np.nan
        if tk in rets.columns:
            vol = annualize_volatility(rets[tk].dropna()) * 100
        # beta regression
        if tk in rets.columns and not factors_df.empty:
            betas, alpha, r2 = compute_factor_betas(rets[tk], factors_df)
            if betas is None:
                beta_dict = {c: np.nan for c in factors_df.columns}
                alpha = np.nan
                r2 = np.nan
            else:
                beta_dict = betas.to_dict()
        else:
            beta_dict = {c: np.nan for c in factors_df.columns}
            alpha = np.nan
            r2 = np.nan
        holdings_metrics.append({
            "ticker": tk,
            "annual_vol_pct": vol,
            "alpha": alpha,
            "r2": r2,
            **{f"beta_{c}": beta_dict.get(c, np.nan) for c in factors_df.columns}
        })

    metrics_df = pd.DataFrame(holdings_metrics).set_index('ticker')
    holdings = holdings.set_index('ticker').join(metrics_df, how='left').reset_index()
    return holdings, rets, factors_df, bench_ret, total_value

def marginal_contribution_to_risk(weights, cov_matrix):
    """
    Compute Marginal Contribution to Risk (MCR) and Percent Contribution (PCR).
    weights: numpy array (n,)
    cov_matrix: numpy array (n,n)
    Returns: MCR array, PCR array
    """
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    if portfolio_vol == 0:
        return np.zeros_like(weights), np.zeros_like(weights)
    # Marginal risk = (Sigma * w)
    mrc = cov_matrix @ weights
    mcr = weights * mrc / portfolio_vol
    pcr = mcr / portfolio_vol
    return mcr, pcr

# ---------- Streamlit UI ----------
st.title("RiskLens — Adaptive Risk & Scenario Simulator (MVP)")
st.markdown("""
Upload a CSV with columns: `ticker, shares, avg_price`.  
Tickers should use Yahoo Finance symbols (e.g., `RELIANCE.NS`, `TCS.NS`, or `AAPL`).  
This MVP uses free data from Yahoo Finance (via `yfinance`).
""")

# sample CSV generator
if st.button("Show sample CSV"):
    sample = pd.DataFrame({
        "ticker": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "shares": [10, 5, 12, 6],
        "avg_price": [2200.0, 3200.0, 1500.0, 1600.0]
    })
    st.code(sample.to_csv(index=False))

uploaded = st.file_uploader("Upload your holdings CSV", type=["csv"])
st.sidebar.header("Settings")
bench_symbol = st.sidebar.text_input("Benchmark (Yahoo ticker)", value="^NSEI")
hist_period = st.sidebar.selectbox("History period for betas (yfinance)", ["1y", "2y", "3y"], index=0)
trading_days = st.sidebar.number_input("Trading days per year", value=252)

if uploaded is None:
    st.info("Upload a CSV to run the analysis (or click 'Show sample CSV').")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()

# Normalize & validate
df.columns = [c.strip().lower() for c in df.columns]
required = {"ticker", "shares", "avg_price"}
if not required.issubset(set(df.columns)):
    st.error(f"CSV must contain columns: {required}. Found columns: {list(df.columns)}")
    st.stop()

df = df[['ticker', 'shares', 'avg_price']].copy()
df['ticker'] = df['ticker'].astype(str).str.strip()
df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)
df['avg_price'] = pd.to_numeric(df['avg_price'], errors='coerce').fillna(0)

# Prepare tickers list (include factors)
user_tickers = df['ticker'].unique().tolist()
factor_tickers = [bench_symbol, "CL=F", "INR=X"]
#  bench included even if same as a user ticker
all_tickers = list(dict.fromkeys(user_tickers + factor_tickers))

st.sidebar.markdown("**Data fetch**")
st.sidebar.write(f"Tickers fetched: {len(all_tickers)} (including benchmark & factors)")

# Fetch price history
with st.spinner("Fetching price history (yfinance)..."):
    prices = fetch_adjusted_close(all_tickers, period=hist_period, interval="1d")

# Compute core portfolio metrics & betas
with st.spinner("Computing portfolio metrics and factor betas..."):
    holdings, rets, factors_df, bench_ret, total_value = portfolio_metrics(df, prices, bench_symbol)

# Show top-line metrics
st.header("Portfolio summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Market Value", f"₹{total_value:,.2f}")
col2.metric("Number of holdings", f"{len(holdings):d}")
# Portfolio annualized vol: compute from weighted daily returns if available
common_rets = rets[ [c for c in rets.columns if c in user_tickers] ].dropna(how='all')
if not common_rets.empty:
    # align weights to columns
    weights = holdings.set_index('ticker').reindex(common_rets.columns)['weight'].fillna(0).values
    port_daily_ret = (common_rets * weights).sum(axis=1)
    port_ann_vol = annualize_volatility(port_daily_ret, trading_days) * 100
    col3.metric("Portfolio annualized vol (est.)", f"{port_ann_vol:.2f}%")
else:
    port_ann_vol = np.nan
    col3.metric("Portfolio annualized vol (est.)", "N/A")

# Portfolio beta (using regression of portfolio returns vs benchmark if possible)
if not common_rets.empty and bench_symbol in rets.columns:
    try:
        port_beta, port_alpha, port_r2 = compute_factor_betas(port_daily_ret, rets[[bench_symbol]])
        port_beta_val = port_beta.get(bench_symbol, np.nan) if isinstance(port_beta, pd.Series) else np.nan
    except Exception:
        port_beta_val = np.nan
else:
    port_beta_val = np.nan
col4.metric(f"Portfolio beta (vs {bench_symbol})", f"{port_beta_val:.2f}" if not np.isnan(port_beta_val) else "N/A")

# Holdings table
st.subheader("Holdings & per-asset metrics")
display_cols = ['ticker', 'shares', 'avg_price', 'market_price', 'market_value', 'cost_basis', 'unrealized_pnl', 'weight', 'annual_vol_pct']
tbl = holdings.copy()
tbl['market_value'] = tbl['market_value'].map(lambda x: f"₹{x:,.2f}")
tbl['cost_basis'] = tbl['cost_basis'].map(lambda x: f"₹{x:,.2f}")
tbl['unrealized_pnl'] = tbl['unrealized_pnl'].map(lambda x: f"₹{x:,.2f}")
tbl['avg_price'] = tbl['avg_price'].map(lambda x: f"₹{x:,.2f}")
tbl['market_price'] = tbl['market_price'].map(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "N/A")
tbl['weight'] = tbl['weight'].map(lambda x: f"{x:.2%}")
tbl['annual_vol_pct'] = tbl['annual_vol_pct'].map(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
st.dataframe(tbl[display_cols].set_index('ticker'), height=300)

# Risk decomposition: covariance & MCR/PCR
st.subheader("Risk decomposition")
# Build covariance matrix for assets (use historical returns)
cov_matrix = None
mcr = pcr = None
if not common_rets.empty:
    # ensure ordering consistent with holdings
    ordered_cols = common_rets.columns.tolist()
    cov = common_rets.cov().values * trading_days  # annualized covariance
    cov_matrix = cov
    weights_arr = holdings.set_index('ticker').reindex(ordered_cols)['weight'].fillna(0).values
    mcr, pcr = marginal_contribution_to_risk(weights_arr, cov_matrix)
    # Prepare dataframe
    rd = pd.DataFrame({
        "ticker": ordered_cols,
        "weight": weights_arr,
        "MCR": mcr,
        "PCR": pcr
    }).set_index('ticker')
    rd['PCR_pct'] = rd['PCR'] * 100
    # show top contributors
    st.markdown("**Top contributors to portfolio risk (PCR %)**")
    st.dataframe(rd.sort_values('PCR_pct', ascending=False)[['weight', 'PCR_pct']].style.format({"weight":"{:.2%}", "PCR_pct":"{:.2f}"}))
    # Risk wheel (pie) using PCR_pct
    fig_wheel = px.pie(rd.sort_values('PCR_pct', ascending=False).head(10).reset_index(), names='ticker', values='PCR_pct', title='Top risk contributors (by % of total risk)')
    st.plotly_chart(fig_wheel, use_container_width=True)
else:
    st.info("Not enough historical returns to compute risk decomposition.")

# Scenario simulator
st.subheader("Scenario simulator — apply shocks and see projected impact")
st.markdown("Set shocks for Market (NIFTY), Crude Oil (CL=F) and USD/INR. Values represent *instantaneous percent change*.")

col_a, col_b, col_c = st.columns(3)
market_shock = col_a.slider(f"{bench_symbol} shock (%)", -15.0, 15.0,  -5.0, step=0.5)
oil_shock = col_b.slider("Crude oil shock (%)", -30.0, 30.0, 10.0, step=0.5)
fx_shock = col_c.slider("USDINR shock (%)", -10.0, 10.0, 2.0, step=0.1)

# Build factor betas matrix: for each asset, get beta_market, beta_CL=F, beta_INR=X
factor_cols = [c for c in prices.columns if c in factor_tickers] if 'prices' in locals() else []
factor_names = [bench_symbol, "CL=F", "INR=X"]
shock_map = {bench_symbol: market_shock/100.0, "CL=F": oil_shock/100.0, "INR=X": fx_shock/100.0}

# For each asset, compute projected return via betas
proj_rows = []
if not factors_df.empty:
    # factors_df columns are exactly the factor tickers available
    for idx, row in holdings.set_index('ticker').iterrows():
        tk = idx
        weight = row['weight']
        # read betas from holdings columns (names like beta_^NSEI / beta_CL=F / beta_INR=X)
        # Our earlier compute named beta columns as beta_<factor>
        projected_ret = 0.0
        betas_used = {}
        for f in factor_names:
            colname = f"beta_{f}" if f in holdings.columns else None
            # column names come from factors_df columns; safe method:
            matching_beta_cols = [c for c in holdings.columns if c.startswith("beta_") and c.endswith(f)]
            if f in holdings.columns:
                beta_val = row.get(f"beta_{f}", np.nan)
            else:
                # fallback: search available beta_ columns for factor matching end
                candidates = [c for c in holdings.columns if c.startswith("beta_")]
                beta_val = np.nan
                for c in candidates:
                    if c.replace("beta_", "") == f:
                        beta_val = row.get(c, np.nan)
                        break
            if pd.isna(beta_val):
                beta_val = 0.0
            betas_used[f] = beta_val
            projected_ret += beta_val * shock_map.get(f, 0.0)
        proj_rows.append({
            "ticker": tk,
            "weight": weight,
            "proj_ret_pct": projected_ret * 100,
            **{f"beta_{f}": betas_used[f] for f in factor_names}
        })
else:
    st.info("Not enough factor history to compute betas; scenario projection will use zero factor exposures.")
    for idx, row in holdings.set_index('ticker').iterrows():
        proj_rows.append({
            "ticker": idx,
            "weight": row['weight'],
            "proj_ret_pct": 0.0,
            **{f"beta_{f}": 0.0 for f in factor_names}
        })

proj_df = pd.DataFrame(proj_rows).set_index('ticker')
# Portfolio projected return under shock
proj_portfolio_return = (proj_df['proj_ret_pct'] / 100.0 * proj_df['weight']).sum() * 100  # in percent
st.metric("Projected instantaneous portfolio return (under shocks)", f"{proj_portfolio_return:.2f}%")

# Waterfall: contribution by factor
# For each factor, compute contribution sum(weight * beta_factor * shock)
factor_contribs = {}
for f in factor_names:
    contrib = 0.0
    for idx, row in proj_df.iterrows():
        beta_col = f"beta_{f}"
        beta_val = row.get(beta_col, 0.0)
        contrib += row['weight'] * beta_val * shock_map.get(f, 0.0)
    factor_contribs[f] = contrib * 100  # in pct points

waterfall_df = pd.DataFrame([
    {"factor": f, "contrib_pct": v} for f, v in factor_contribs.items()
])
# Add residual (idiosyncratic) as remaining projected return
residual = proj_portfolio_return - waterfall_df['contrib_pct'].sum()
waterfall_df = pd.concat([waterfall_df, pd.DataFrame([{"factor": "idiosyncratic", "contrib_pct": residual}])], ignore_index=True)

fig_wf = go.Figure()
fig_wf.add_trace(go.Bar(x=waterfall_df['factor'], y=waterfall_df['contrib_pct'], marker_color=px.colors.qualitative.Plotly))
fig_wf.update_layout(title="Projected portfolio return decomposition (pct points)", yaxis_title="Percentage points")
st.plotly_chart(fig_wf, use_container_width=True)

# Show projected changes per holding (table)
st.subheader("Projected per-holding instantaneous returns under shocks")
proj_display = proj_df.copy()
proj_display['weight_pct'] = proj_display['weight'] * 100
proj_display['proj_ret_pct'] = proj_display['proj_ret_pct'].map(lambda x: f"{x:.2f}%")
proj_display['weight_pct'] = proj_display['weight_pct'].map(lambda x: f"{x:.2f}%")
st.dataframe(proj_display[['weight_pct', 'proj_ret_pct']].rename(columns={"weight_pct":"weight", "proj_ret_pct":"proj_return"}))

st.markdown("""
---
### Notes / Limitations
- This MVP uses historic linear factor regression (OLS) to estimate sensitivities. Betas are dependent on the chosen lookback and data quality.
- Data source: Yahoo Finance via `yfinance` (free). Some Indian tickers may occasionally have incomplete data.
- The scenario module applies instantaneous shocks via factor betas — it is a first-order linear approximation and is not a substitute for full simulation or live trading systems.
""")
