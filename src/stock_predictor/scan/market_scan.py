from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from stock_predictor.config import MODELS_DIR
from stock_predictor.data.ingest import download_prices
from stock_predictor.features.engineering import build_training_frame, split_features_target
from stock_predictor.live.settings import has_alpaca_credentials


@dataclass
class UniverseRow:
    yahoo_symbol: str
    broker_symbol: Optional[str]  # e.g., NSE:RELIANCE for Kite
    tradingsymbol: Optional[str]  # e.g., RELIANCE
    exchange: Optional[str]       # e.g., NSE


@dataclass
class ScanConfig:
    universe_csv: Path
    start: str
    end: str
    model_type: str
    threshold: float
    total_equity: float
    per_position_fraction: float  # max fraction of equity per name (e.g., 0.05 => 5%)
    top_k: int
    broker: str  # "alpaca" or "kite"
    use_broker_price: bool


@dataclass
class ScanRowResult:
    yahoo_symbol: str
    broker_symbol: Optional[str]
    predicted_return: Optional[float]
    last_price: Optional[float]
    signal_long: bool
    suggested_allocation: float
    suggested_qty: int
    reason: Optional[str]


def _load_model(yahoo_symbol: str, model_type: str):
    path = MODELS_DIR / f"{yahoo_symbol.upper()}_{model_type}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _latest_features(yahoo_symbol: str, start: str, end: str) -> pd.DataFrame:
    raw = download_prices(yahoo_symbol, start=start, end=end)
    frame = build_training_frame(raw)
    X, _ = split_features_target(frame)
    return X.iloc[[-1]].copy()


def _get_last_price(yahoo_symbol: str, broker_symbol: Optional[str], broker: str, use_broker_price: bool, start: str, end: str) -> Optional[float]:
    try:
        if broker == "alpaca" and use_broker_price and has_alpaca_credentials():
            from ..live.broker_alpaca import AlpacaBroker  # type: ignore
            b = AlpacaBroker()
            return float(b.get_last_price(yahoo_symbol))
        if broker == "kite" and use_broker_price and broker_symbol:
            from ..live.settings import has_kite_credentials
            if has_kite_credentials():
                from ..live.broker_kite import KiteBroker  # type: ignore
                b = KiteBroker()
                return float(b.get_last_price(broker_symbol))
        # fallback to Yahoo close (use scan window)
        df = download_prices(yahoo_symbol, start=start, end=end)
        return float(df["Close"].iloc[-1])
    except Exception:
        try:
            df = download_prices(yahoo_symbol, start=start, end=end)
            return float(df["Close"].iloc[-1])
        except Exception:
            return None


def load_universe(csv_path: Path) -> List[UniverseRow]:
    rows: List[UniverseRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                UniverseRow(
                    yahoo_symbol=r.get("yahoo_symbol", "").strip(),
                    broker_symbol=r.get("broker_symbol", "").strip() or None,
                    tradingsymbol=r.get("tradingsymbol", "").strip() or None,
                    exchange=r.get("exchange", "").strip() or None,
                )
            )
    return rows


def scan_market(cfg: ScanConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    universe = load_universe(cfg.universe_csv)
    results: List[ScanRowResult] = []

    for u in universe:
        model_payload = _load_model(u.yahoo_symbol, cfg.model_type)
        if model_payload is None:
            results.append(
                ScanRowResult(
                    yahoo_symbol=u.yahoo_symbol,
                    broker_symbol=u.broker_symbol,
                    predicted_return=None,
                    last_price=_get_last_price(u.yahoo_symbol, u.broker_symbol, cfg.broker, cfg.use_broker_price, cfg.start, cfg.end),
                    signal_long=False,
                    suggested_allocation=0.0,
                    suggested_qty=0,
                    reason="model_not_found",
                )
            )
            continue

        X_latest = _latest_features(u.yahoo_symbol, cfg.start, cfg.end)
        pred = float(model_payload["model"].predict(X_latest)[0])
        last_price = _get_last_price(u.yahoo_symbol, u.broker_symbol, cfg.broker, cfg.use_broker_price, cfg.start, cfg.end)
        signal = bool(pred > cfg.threshold)

        results.append(
            ScanRowResult(
                yahoo_symbol=u.yahoo_symbol,
                broker_symbol=u.broker_symbol,
                predicted_return=pred,
                last_price=last_price,
                signal_long=signal,
                suggested_allocation=0.0,  # to be filled later
                suggested_qty=0,
                reason=None,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results])

    # Determine investable candidates
    candidates = df[(df["signal_long"] == True) & df["last_price"].notna()].copy()
    candidates = candidates.sort_values("predicted_return", ascending=False)

    # Size per name
    max_per_name = cfg.total_equity * cfg.per_position_fraction
    if cfg.top_k > 0 and len(candidates) > cfg.top_k:
        candidates = candidates.head(cfg.top_k)

    # Equal-weight among selected up to max_per_name
    if len(candidates) > 0:
        per_name_budget = min(max_per_name, cfg.total_equity / len(candidates))
        qty = np.floor(per_name_budget / candidates["last_price"]).astype(int)
        alloc = qty * candidates["last_price"]
        candidates.loc[:, "suggested_qty"] = qty
        candidates.loc[:, "suggested_allocation"] = alloc

    # Merge sizing back
    df = df.merge(
        candidates[["yahoo_symbol", "suggested_qty", "suggested_allocation"]],
        on="yahoo_symbol",
        how="left",
        suffixes=("", "_cand"),
    )
    df["suggested_qty"] = df["suggested_qty"].fillna(0).astype(int)
    df["suggested_allocation"] = df["suggested_allocation"].fillna(0.0)

    summary = pd.DataFrame({
        "total_candidates": [int((df["signal_long"] == True).sum())],
        "selected": [int((df["suggested_qty"] > 0).sum())],
        "capital_used": [float(df["suggested_allocation"].sum())],
    })

    return df.sort_values(["signal_long", "predicted_return"], ascending=[False, False]), summary
