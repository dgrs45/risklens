import yfinance as yf
import pandas as pd

def fetch_adjusted_close(tickers, period="1y", interval="1d"):
    """
    Fetch adjusted close prices for a list of tickers via yfinance.
    Returns DataFrame of prices with datetime index.
    """
    df = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        out = {tk: df[tk]["Adj Close"].rename(tk) for tk in tickers if tk in df.columns.levels[0]}
        prices = pd.DataFrame(out)
    else:
        prices = df
    prices = prices.dropna(axis=1, how="all").sort_index()
    return prices

def fetch_latest_price(ticker):
    """
    Return most recent adjusted close for a single ticker.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="5d", interval="1d", auto_adjust=True)
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    return None
