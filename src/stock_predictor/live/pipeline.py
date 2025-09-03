from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from ..data.ingest import download_prices
from ..features.engineering import build_training_frame, split_features_target
from .settings import has_alpaca_credentials


@dataclass
class LiveConfig:
    ticker: str
    start: str
    end: str
    model_type: str
    threshold: float
    risk_fraction: float  # fraction of account equity to allocate when long
    dry_run: bool
    simulate_equity: float = 100000.0
    broker: str = "alpaca"  # or "kite"
    data_symbol: Optional[str] = None  # e.g., 'NSE:RELIANCE' for Kite pricing


@dataclass
class LiveDecision:
    predicted_return: float
    signal_long: bool
    qty: float
    action: str  # buy/close/hold
    price: float


def load_saved_model(ticker: str, model_type: str):
    path = MODELS_DIR / f"{ticker.upper()}_{model_type}.joblib"
    payload = joblib.load(path)
    return payload


essential_feature_cols_cache = None


def latest_features_for_prediction(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = download_prices(ticker=ticker, start=start, end=end)
    frame = build_training_frame(raw)
    X, y = split_features_target(frame)
    return X.iloc[[-1]].copy()


def decide_and_size(
    predicted_next_return: float,
    account_equity: float,
    last_price: float,
    threshold: float,
    risk_fraction: float,
) -> (bool, float, str):
    if predicted_next_return > threshold:
        allocation = max(0.0, min(1.0, risk_fraction)) * account_equity
        qty = max(0.0, np.floor(allocation / last_price))
        return True, float(qty), "buy" if qty > 0 else "hold"
    else:
        return False, 0.0, "close"


def run_live(config: LiveConfig) -> LiveDecision:
    payload = load_saved_model(config.ticker, config.model_type)
    model = payload["model"]

    X_latest = latest_features_for_prediction(config.ticker, config.start, config.end)
    pred = float(model.predict(X_latest)[0])

    # Determine equity and price sources depending on credentials and dry-run
    if not config.dry_run:
        if config.broker == "alpaca" and has_alpaca_credentials():
            # Import only when needed to avoid optional dependency issues
            from .broker_alpaca import AlpacaBroker  # type: ignore
            broker = AlpacaBroker()
            account = broker.get_account()
            equity = float(account.equity)
            last_price = broker.get_last_price(config.ticker)
        elif config.broker == "kite":
            from .settings import has_kite_credentials
            if has_kite_credentials():
                from .broker_kite import KiteBroker  # type: ignore
                broker = KiteBroker()
                equity = float(broker.get_net_equity())
                symbol = config.data_symbol or config.ticker  # expect 'NSE:SYMBOL' here
                last_price = float(broker.get_last_price(symbol))
            else:
                broker = None  # type: ignore
                frame = download_prices(config.ticker, start=config.start, end=config.end)
                last_price = float(frame["Close"].iloc[-1])
                equity = float(config.simulate_equity)
        else:
            broker = None  # type: ignore
            frame = download_prices(config.ticker, start=config.start, end=config.end)
            last_price = float(frame["Close"].iloc[-1])
            equity = float(config.simulate_equity)
    else:
        # Fallback to simulated mode using most recent close
        frame = download_prices(config.ticker, start=config.start, end=config.end)
        last_price = float(frame["Close"].iloc[-1])
        equity = float(config.simulate_equity)
        broker = None  # type: ignore

    signal_long, qty, action = decide_and_size(
        predicted_next_return=pred,
        account_equity=equity,
        last_price=last_price,
        threshold=config.threshold,
        risk_fraction=config.risk_fraction,
    )

    # Execute
    if not config.dry_run and broker is not None:
        if config.broker == "alpaca":
            current_qty = broker.get_position_qty(config.ticker)
            if action == "buy" and qty > 0:
                delta = max(0.0, qty - current_qty)
                if delta > 0:
                    broker.submit_market_order(config.ticker, delta, side="buy")
            elif action == "close" and current_qty > 0:
                broker.close_position(config.ticker)
        elif config.broker == "kite":
            # Expect data_symbol like 'NSE:RELIANCE' => exchange 'NSE', tradingsymbol 'RELIANCE'
            symbol = config.data_symbol or config.ticker
            if ":" in symbol:
                exchange, tradingsymbol = symbol.split(":", 1)
            else:
                exchange, tradingsymbol = "NSE", symbol
            current_qty = broker.get_position_qty(tradingsymbol)
            if action == "buy" and qty > 0:
                delta = max(0, int(qty) - int(current_qty))
                if delta > 0:
                    broker.submit_market_order(exchange, tradingsymbol, delta, side="buy")
            elif action == "close" and current_qty > 0:
                broker.close_position(exchange, tradingsymbol)

    return LiveDecision(
        predicted_return=pred,
        signal_long=signal_long,
        qty=qty,
        action=action,
        price=last_price,
    )
