# risklens_core/risk_engine.py
import numpy as np
import pandas as pd
import statsmodels.api as sm

def compute_returns(price_series: pd.Series) -> pd.Series:
    """Simple percent returns series."""
    return price_series.pct_change().dropna()

def annualize_volatility(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized volatility (decimal, e.g. 0.15 for 15%)."""
    if daily_returns.empty:
        return float("nan")
    return float(daily_returns.std() * (trading_days ** 0.5))

def compute_factor_betas(stock_returns: pd.Series, factors_df: pd.DataFrame):
    """
    OLS regression of stock_returns on columns in factors_df.
    Returns: (betas: pd.Series, alpha: float, r2: float)
    If not enough data, returns (None, None, None).
    """
    df = pd.concat([stock_returns, factors_df], axis=1).dropna()
    if df.shape[0] < 30:
        return None, None, None
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    params = model.params
    alpha = float(params.get("const", np.nan))
    betas = params.drop("const", errors="ignore")
    return betas, alpha, float(model.rsquared)

def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Return portfolio variance (annualized if cov_matrix is annualized)."""
    return float(weights.T @ cov_matrix @ weights)

def marginal_contribution_to_risk(weights: np.ndarray, cov_matrix: np.ndarray):
    """
    Compute Marginal Contribution to Risk (MCR) and Percent Contribution to Risk (PCR).
    weights: numpy array (n,)
    cov_matrix: numpy array (n,n) (should be annualized)
    Returns: mcr (n,), pcr (n,) where pcr sums to portfolio_vol
    """
    weights = np.asarray(weights)
    port_var = float(weights.T @ cov_matrix @ weights)
    port_vol = np.sqrt(port_var) if port_var >= 0 else np.nan
    if port_vol == 0 or np.isnan(port_vol):
        mcr = np.zeros_like(weights)
        pcr = np.zeros_like(weights)
        return mcr, pcr
    marginal = cov_matrix @ weights             # (n,)
    mcr = (weights * marginal) / port_vol       # marginal contribution
    pcr = mcr / port_vol                        # percent contribution (sums to 1)
    return mcr, pcr
