# Multi-Asset Portfolio Risk Dashboard

A professional risk analytics dashboard built with Python and Streamlit.

## Features
- Live price data via yfinance
- VaR: Historical, Parametric, Monte Carlo
- CVaR (Expected Shortfall)
- Sharpe & Sortino Ratios
- Max Drawdown
- Stress Testing (COVID, 2022 Rate Hike)
- Asset Correlation Heatmap
- Individual Asset Risk Breakdown

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app"
4. Select your repo → branch → set main file as `app.py`
5. Click Deploy

Your app will be live at:
`https://your-username-your-repo-app-py-xxxx.streamlit.app`

## Deploy to Render (Free)

1. Push to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy

## Project Structure

```
risk_dashboard/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Stack
- Python 3.10+
- Streamlit
- Plotly
- yfinance
- NumPy / Pandas
