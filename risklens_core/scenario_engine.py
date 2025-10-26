# risklens_core/scenario_engine.py
import pandas as pd

def simulate_scenario(holdings_df: pd.DataFrame, shock_map: dict, factor_names: list):
    """
    holdings_df: DataFrame with index=ticker and columns includes:
                 'weight' and 'beta_<factor>' for each factor in factor_names.
    shock_map: dict mapping factor name -> shock in decimal (e.g. -0.05 for -5%)
    factor_names: ordered list of factor names used as suffixes in holdings_df's beta columns.
    Returns:
       holdings_out: copy of holdings_df with 'proj_ret_pct' column (percent)
       port_proj_return_pct: scalar projected portfolio return in percent
    """
    h = holdings_df.copy()
    proj_vals = []
    for idx, row in h.iterrows():
        total = 0.0
        for f in factor_names:
            beta_col = f"beta_{f}"
            beta_val = row.get(beta_col, 0.0)
            if pd.isna(beta_val):
                beta_val = 0.0
            shock = shock_map.get(f, 0.0)
            total += beta_val * shock
        proj_vals.append(total * 100.0)  # percent
    h["proj_ret_pct"] = proj_vals
    # portfolio projected return (percent) = sum(weight * proj_ret_pct) / 100
    port_proj_return_pct = float((h["proj_ret_pct"] * h["weight"]).sum())
    return h, port_proj_return_pct
