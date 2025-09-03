from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..features.engineering import build_training_frame, split_features_target
from ..data.ingest import download_prices
from ..models.train import load_model


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    buy_and_hold_curve: pd.Series
    summary: dict


def backtest(
    model_path: Path,
    ticker: str,
    start: str,
    end: str,
    threshold: float = 0.0,
    transaction_cost: float = 0.0005,
) -> BacktestResult:
    payload = load_model(model_path)
    model = payload["model"]

    raw = download_prices(ticker=ticker, start=start, end=end)
    frame = build_training_frame(raw)
    X, y = split_features_target(frame)

    preds = pd.Series(model.predict(X), index=X.index, name="predicted_return")

    # Strategy: long if predicted next-day return > threshold
    signal = (preds > threshold).astype(int)

    # Realized next-day return
    realized = y.copy()

    strategy_return = signal.shift(1).fillna(0) * realized - np.abs(signal.diff().fillna(0)) * transaction_cost
    equity_curve = (1 + strategy_return).cumprod()

    buy_and_hold_curve = (1 + realized).cumprod()

    sharpe = np.sqrt(252) * strategy_return.mean() / (strategy_return.std() + 1e-9)
    summary = {
        "final_equity": float(equity_curve.iloc[-1]),
        "final_bh": float(buy_and_hold_curve.iloc[-1]),
        "annualized_return": float((equity_curve.iloc[-1]) ** (252 / len(equity_curve)) - 1),
        "sharpe": float(sharpe),
        "win_rate": float((strategy_return > 0).mean()),
    }

    return BacktestResult(equity_curve=equity_curve, buy_and_hold_curve=buy_and_hold_curve, summary=summary)
