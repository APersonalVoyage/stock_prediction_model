import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.stock_predictor.models.train import train_ticker, load_model
from src.stock_predictor.models.evaluate import evaluate_saved_model
from src.stock_predictor.backtest.backtest import backtest
from src.stock_predictor.config import MODELS_DIR

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Prediction & Backtest")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="AAPL")
with col2:
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
with col3:
    end = st.date_input("End", value=pd.to_datetime("2025-01-01")).strftime("%Y-%m-%d")

model_type = st.selectbox("Model", ["rf", "xgb"], index=0)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Train Model"):
        with st.spinner("Training..."):
            result = train_ticker(ticker=ticker, start=start, end=end, model_type=model_type, save_model=True)
            st.success(f"Model saved to {result.model_path}")
            st.json(result.metrics)

with col_b:
    threshold = st.number_input("Backtest Threshold (predicted next-day return)", value=0.0, step=0.0005, format="%.4f")
    transaction_cost = st.number_input("Transaction Cost per trade", value=0.0005, step=0.0001, format="%.4f")
    if st.button("Run Backtest"):
        model_path = MODELS_DIR / f"{ticker.upper()}_{model_type}.joblib"
        if not model_path.exists():
            st.warning("Model not found. Train first.")
        else:
            with st.spinner("Backtesting..."):
                result = backtest(
                    model_path=model_path,
                    ticker=ticker,
                    start=start,
                    end=end,
                    threshold=threshold,
                    transaction_cost=transaction_cost,
                )
                fig, ax = plt.subplots(figsize=(10, 4))
                result.equity_curve.plot(ax=ax, label="Strategy")
                result.buy_and_hold_curve.plot(ax=ax, label="Buy & Hold")
                ax.set_title("Equity Curve")
                ax.legend()
                st.pyplot(fig)

                st.subheader("Backtest Summary")
                st.json(result.summary)
