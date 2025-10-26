#need to check beta calculation - not sure why it is not working
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# core modules (must exist in risklens_core/)
import risklens_core.data_engine as data_engine
import risklens_core.risk_engine as risk_engine
import risklens_core.scenario_engine as scenario_engine
import risklens_core.helper as helper

st.set_page_config(page_title="RiskLens — Portfolio Risk & Scenario Simulator", layout="wide")
st.title("RiskLens — Adaptive Risk & Scenario Simulator (MVP)")
st.markdown("Upload a CSV with columns: `ticker, shares, avg_price`. Use Yahoo tickers (e.g., `RELIANCE.NS`).")

# ---------------- Sidebar: settings & input source ----------------
st.sidebar.header("Settings")
bench_symbol = st.sidebar.text_input("Benchmark (Yahoo ticker)", value="^NSEI")
hist_period = st.sidebar.selectbox("History period for betas", ["1y", "2y", "3y"], index=0)
trading_days = int(st.sidebar.number_input("Trading days / year", value=252))

st.sidebar.markdown("---")
st.sidebar.markdown("**Input**")
use_sample = st.sidebar.checkbox("Use bundled sample CSV (data/sample_portfolio.csv)", value=True)
uploaded = st.file_uploader("Upload your holdings CSV", type=["csv"])

# ---------------- Load holdings (uploaded or sample) ----------------
if uploaded is not None:
    try:
        holdings_df = helper.load_portfolio_csv(uploaded)
    except Exception as e:
        st.error(f"Error parsing uploaded CSV: {e}")
        st.stop()
else:
    if use_sample:
        sample_path = "data/sample_portfolio.csv"
        try:
            with open(sample_path, "rb") as f:
                holdings_df = helper.load_portfolio_csv(f)
            st.info(f"Loaded sample: {sample_path}")
        except FileNotFoundError:
            st.error(f"Sample CSV not found at {sample_path}. Please upload a CSV.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading sample CSV: {e}")
            st.stop()
    else:
        st.info("Upload a CSV or enable the 'Use bundled sample CSV' toggle in the sidebar.")
        st.stop()

# ---------------- Prepare tickers & fetch prices ----------------
user_tickers = holdings_df["ticker"].unique().tolist()
factor_tickers = [bench_symbol, "CL=F", "INR=X"]
all_tickers = list(dict.fromkeys(user_tickers + factor_tickers))

st.sidebar.markdown(f"Fetching {len(all_tickers)} tickers (free data via yfinance)")

with st.spinner("Fetching price history..."):
    prices = data_engine.fetch_adjusted_close(all_tickers, period=hist_period, interval="1d")
    # prices is DataFrame columns = tickers (only those available)

# ---------------- Assemble holdings metrics ----------------
def assemble_holdings(holdings, prices_df, bench_sym):
    holdings = holdings.copy()
    # latest prices (prefer historical series last value, fallback to single fetch)
    latest_prices = {}
    for tk in holdings["ticker"]:
        if tk in prices_df.columns and not prices_df[tk].dropna().empty:
            latest_prices[tk] = float(prices_df[tk].dropna().iloc[-1])
        else:
            latest_prices[tk] = data_engine.fetch_latest_price(tk)
    holdings["market_price"] = holdings["ticker"].map(latest_prices)
    holdings["market_value"] = holdings["market_price"] * holdings["shares"]
    holdings["cost_basis"] = holdings["avg_price"] * holdings["shares"]
    holdings["unrealized_pnl"] = holdings["market_value"] - holdings["cost_basis"]
    total_value = float(holdings["market_value"].sum())
    holdings["weight"] = holdings["market_value"] / total_value if total_value != 0 else 0.0

    # returns & factors
    rets = prices_df[user_tickers].pct_change().dropna(how="all") if any(c in prices_df.columns for c in user_tickers) else pd.DataFrame()
    factor_cols = [c for c in prices_df.columns if c in factor_tickers]
    factors_df = prices_df[factor_cols].pct_change().dropna() if factor_cols else pd.DataFrame()

    rows = []
    for _, r in holdings.iterrows():
        tk = r["ticker"]
        vol = np.nan
        if tk in rets.columns:
            vol = risk_engine.annualize_volatility(rets[tk], trading_days) * 100
        if tk in rets.columns and not factors_df.empty:
            betas, alpha, r2 = risk_engine.compute_factor_betas(rets[tk], factors_df)
            if betas is None:
                beta_dict = {f: np.nan for f in factors_df.columns}
            else:
                beta_dict = betas.to_dict()
        else:
            beta_dict = {f: np.nan for f in factors_df.columns}
            alpha = np.nan
            r2 = np.nan

        rowd = {
            "ticker": tk,
            "shares": r["shares"],
            "avg_price": r["avg_price"],
            "market_price": r["market_price"],
            "market_value": r["market_value"],
            "cost_basis": r["cost_basis"],
            "unrealized_pnl": r["unrealized_pnl"],
            "weight": r["weight"],
            "annual_vol_pct": vol,
            "alpha": alpha,
            "r2": r2
        }
        for f in factors_df.columns:
            rowd[f"beta_{f}"] = beta_dict.get(f, np.nan)
        rows.append(rowd)

    out = pd.DataFrame(rows).set_index("ticker")
    return out, rets, factors_df, total_value

with st.spinner("Computing holdings metrics..."):
    holdings_out, rets, factors_df, portfolio_value = assemble_holdings(holdings_df, prices, bench_symbol)

# ---------------- Top-line metrics ----------------
st.header("Portfolio summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Market Value", helper.format_rupee(portfolio_value))
c2.metric("Number of holdings", str(len(holdings_out)))

if not rets.empty:
    ordered = [c for c in rets.columns if c in holdings_out.index]
    if ordered:
        cov = rets[ordered].cov().values * trading_days
        weights = holdings_out.reindex(ordered)["weight"].fillna(0).values
        port_var = risk_engine.portfolio_variance(weights, cov)
        port_vol_pct = np.sqrt(port_var) * 100
        c3.metric("Portfolio annualized vol (est.)", f"{port_vol_pct:.2f}%")
    else:
        c3.metric("Portfolio annualized vol (est.)", "N/A")
else:
    c3.metric("Portfolio annualized vol (est.)", "N/A")

# portfolio beta vs benchmark (regress portfolio returns on bench)
if not rets.empty and bench_symbol in rets.columns:
    # construct portfolio daily returns (value-weighted)
    avail = [c for c in rets.columns if c in holdings_out.index]
    if avail:
        port_daily_ret = (rets[avail] * holdings_out.reindex(avail)["weight"]).sum(axis=1)
        betas_port, alpha_p, r2_p = risk_engine.compute_factor_betas(port_daily_ret, rets[[bench_symbol]])
        port_beta_val = betas_port.get(bench_symbol, np.nan) if betas_port is not None else np.nan
        c4.metric(f"Portfolio beta (vs {bench_symbol})", f"{port_beta_val:.2f}" if not np.isnan(port_beta_val) else "N/A")
    else:
        c4.metric(f"Portfolio beta (vs {bench_symbol})", "N/A")
else:
    c4.metric(f"Portfolio beta (vs {bench_symbol})", "N/A")

# ---------------- Holdings table ----------------
st.subheader("Holdings & metrics")
display = holdings_out.reset_index().copy()
display["market_value"] = display["market_value"].map(lambda x: helper.format_rupee(x))
display["cost_basis"] = display["cost_basis"].map(lambda x: helper.format_rupee(x))
display["unrealized_pnl"] = display["unrealized_pnl"].map(lambda x: helper.format_rupee(x))
display["weight"] = display["weight"].map(lambda x: f"{x:.2%}")
display["market_price"] = display["market_price"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
st.dataframe(display[["ticker", "shares", "avg_price", "market_price", "market_value", "unrealized_pnl", "weight", "annual_vol_pct"]].set_index("ticker"), height=320)

# ---------------- Risk decomposition ----------------
st.subheader("Risk decomposition (MCR / PCR)")
if not rets.empty:
    ordered_cols = [c for c in rets.columns if c in holdings_out.index]
    if ordered_cols:
        cov_matrix = rets[ordered_cols].cov().values * trading_days
        weights_arr = holdings_out.reindex(ordered_cols)["weight"].fillna(0).values
        mcr, pcr = risk_engine.marginal_contribution_to_risk(weights_arr, cov_matrix)
        rd = pd.DataFrame({"ticker": ordered_cols, "weight": weights_arr, "MCR": mcr, "PCR": pcr}).set_index("ticker")
        rd["PCR_pct"] = rd["PCR"] * 100
        st.dataframe(rd.sort_values("PCR_pct", ascending=False)[["weight", "PCR_pct"]].style.format({"weight": "{:.2%}", "PCR_pct": "{:.2f}"}))
        fig = px.pie(rd.reset_index().head(10), names="ticker", values="PCR_pct", title="Top risk contributors (PCR %)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough overlapping historical data to compute risk decomposition.")
else:
    st.info("Not enough historical returns to compute risk decomposition.")

# ---------------- Scenario simulator ----------------
st.subheader("Scenario simulator")
st.markdown("Apply instantaneous shocks to Market (NIFTY), Crude Oil (CL=F), and USD/INR. Values are percent changes.")

col_a, col_b, col_c = st.columns(3)
market_shock = col_a.slider(f"{bench_symbol} shock (%)", -15.0, 15.0, -5.0, step=0.5)
oil_shock = col_b.slider("Crude oil shock (%)", -30.0, 30.0, 10.0, step=0.5)
fx_shock = col_c.slider("USDINR shock (%)", -10.0, 10.0, 2.0, step=0.1)

factor_names = [bench_symbol, "CL=F", "INR=X"]
shock_map = {bench_symbol: market_shock / 100.0, "CL=F": oil_shock / 100.0, "INR=X": fx_shock / 100.0}

proj_input = holdings_out.copy()
# ensure beta columns exist
for f in factor_names:
    col = f"beta_{f}"
    if col not in proj_input.columns:
        proj_input[col] = 0.0

proj_holdings, port_proj_pct = scenario_engine.simulate_scenario(proj_input, shock_map, factor_names)
st.metric("Projected instantaneous portfolio return (under shocks)", f"{port_proj_pct:.2f}%")

# waterfall
contribs = []
for f in factor_names:
    contrib = float((proj_input[f"beta_{f}"] * proj_input["weight"] * (shock_map[f] * 100)).sum())
    contribs.append({"factor": f, "contrib_pct": contrib})
residual = float(port_proj_pct - sum([c["contrib_pct"] for c in contribs]))
contribs.append({"factor": "idiosyncratic", "contrib_pct": residual})
wf = pd.DataFrame(contribs)
fig_wf = go.Figure([go.Bar(x=wf["factor"], y=wf["contrib_pct"])])
fig_wf.update_layout(title="Projected portfolio return decomposition (pct points)", yaxis_title="Percentage points")
st.plotly_chart(fig_wf, use_container_width=True)

# per-holding projection
st.subheader("Projected per-holding returns (instantaneous)")
disp_proj = proj_holdings.reset_index().copy()
disp_proj["weight"] = disp_proj["weight"].map(lambda x: f"{x:.2%}")
disp_proj["proj_ret_pct"] = disp_proj["proj_ret_pct"].map(lambda x: f"{x:.2f}%")
st.dataframe(disp_proj[["ticker", "weight", "proj_ret_pct"]].set_index("ticker"), height=300)

st.markdown(
    """
**Notes & limitations**

- Betas are estimated via OLS on historical daily returns; results depend on the lookback horizon and data quality.
- This is a first-order stress approximation (linear). For large shocks, non-linear effects and liquidity impacts are not modeled.
- Data source: free Yahoo Finance via `yfinance`.
"""
)
