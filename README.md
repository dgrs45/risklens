# RiskLens — Adaptive Portfolio Risk & Scenario Simulator

**RiskLens** is a Streamlit dashboard that helps investors analyze the risk, volatility, and macro sensitivity of their Indian equity portfolios — using only free data from Yahoo Finance.

---

### What It Does
- Upload a CSV of your holdings (`ticker, shares, avg_price`)
- Fetches real-time market prices from Yahoo Finance  
- Calculates portfolio value, returns, volatility, and betas  
- Decomposes risk contributions by holding  
- Runs “what-if” macro scenarios (Market / Crude Oil / USD-INR)  
- Shows projected stress impacts with interactive visuals

---

### Quick Start

```bash
git clone https://github.com/<your-username>/risklens.git
cd risklens
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
