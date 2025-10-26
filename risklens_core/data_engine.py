import yfinance as yf
import pandas as pd

def fetch_adjusted_close(tickers, period="1y", interval="1d"):
    """
    Fetch adjusted close prices for a list of tickers via yfinance.
    It attempts to use 'Adj Close', but falls back to 'Close' if 'Adj Close' is missing
    (common for FX/commodity tickers).
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
    
    out = {}
    
    # Check if the result is a MultiIndex DataFrame (for multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        
        # Iterate through tickers to safely extract prices
        for tk in tickers:
            if tk in df.columns.levels[0]:
                ticker_df = df[tk]
                
                # Logic to check for 'Adj Close' and fall back to 'Close'
                if "Adj Close" in ticker_df.columns:
                    series = ticker_df["Adj Close"].rename(tk)
                elif "Close" in ticker_df.columns:
                    series = ticker_df["Close"].rename(tk)
                else:
                    # Skip the ticker if neither key is found
                    continue
                    
                out[tk] = series
        
        prices = pd.DataFrame(out)
        
    else:
        # Handling the case for a single ticker where MultiIndex isn't used.
        # Check if 'Adj Close' is available, otherwise use 'Close'.
        if "Adj Close" in df.columns:
            prices = df["Adj Close"].to_frame(name=tickers[0])
        elif "Close" in df.columns:
            prices = df["Close"].to_frame(name=tickers[0])
        else:
            prices = pd.DataFrame() # Return empty if no price column found


    prices = prices.dropna(axis=1, how="all").sort_index()
    return prices

def fetch_latest_price(ticker):
    """
    Return most recent adjusted close for a single ticker.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="5d", interval="1d", auto_adjust=True)
    if not hist.empty:
        # 'auto_adjust=True' means the 'Close' column already contains the final adjusted price.
        return float(hist["Close"].iloc[-1])
    return None