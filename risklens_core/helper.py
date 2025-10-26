# risklens_core/utils.py
import pandas as pd

def load_portfolio_csv(file_like):
    """
    Read CSV uploaded by user. Expect columns: ticker, shares, avg_price
    Returns cleaned DataFrame with those columns and types normalized.
    """
    df = pd.read_csv(file_like)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"ticker", "shares", "avg_price"}
    if not required.issubset(set(df.columns)):
        missing = required.difference(set(df.columns))
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df[['ticker', 'shares', 'avg_price']].copy()
    df['ticker'] = df['ticker'].astype(str).str.strip()
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)
    df['avg_price'] = pd.to_numeric(df['avg_price'], errors='coerce').fillna(0)
    return df

def format_rupee(x):
    """Format numbers as Indian rupees."""
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "N/A"
