from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # Trend indicators
    df["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["ema_10"] = EMAIndicator(close, window=10).ema_indicator()
    df["ema_20"] = EMAIndicator(close, window=20).ema_indicator()

    # Momentum
    df["rsi_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Volatility
    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / close

    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Targets: next-day return
    df["target_next_1d_return"] = df["Close"].pct_change().shift(-1)

    # Lags of features (to avoid leakage)
    lag_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "sma_10",
        "sma_20",
        "ema_10",
        "ema_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_high",
        "bb_low",
        "bb_width",
        "return_1d",
        "return_5d",
        "return_10d",
    ]
    for col in lag_cols:
        df[f"{col}_lag1"] = df[col].shift(1)

    return df


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_technical_indicators(df)
    df = add_return_features(df)

    # Drop rows with NaNs produced by indicators/lags and the last row due to shifted target
    df = df.dropna()
    return df


def split_features_target(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.endswith("_lag1") or c in [
        "bb_width",
    ]]
    X = df[feature_cols].copy()
    y = df["target_next_1d_return"].copy()
    return X, y
