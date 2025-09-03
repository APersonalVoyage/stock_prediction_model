from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class OrderPlan:
    broker_symbol: Optional[str]
    exchange: Optional[str]
    tradingsymbol: Optional[str]
    current_qty: int
    target_qty: int
    action: str  # buy/sell/hold/skip
    delta: int
    message: str


@dataclass
class RebalanceSummary:
    planned: int
    submitted: int
    skipped: int


def _parse_broker_symbol(broker_symbol: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not broker_symbol:
        return None, None
    if ":" in broker_symbol:
        ex, ts = broker_symbol.split(":", 1)
        return ex, ts
    return None, broker_symbol


def _get_kite_broker():
    from ..live.broker_kite import KiteBroker  # lazy import
    return KiteBroker()


def rebalance_from_scan(
    scan_csv: Path,
    broker: str = "kite",
    dry_run: bool = True,
    max_per_order_qty: Optional[int] = None,
) -> Tuple[RebalanceSummary, List[OrderPlan]]:
    df = pd.read_csv(scan_csv)
    plans: List[OrderPlan] = []

    if broker != "kite":
        raise ValueError("Currently only broker='kite' is supported for rebalance")

    kb = _get_kite_broker() if not dry_run else None

    for _, row in df.iterrows():
        broker_symbol = row.get("broker_symbol") if pd.notna(row.get("broker_symbol")) else None
        suggested_qty = int(row.get("suggested_qty") or 0)
        signal_long = bool(row.get("signal_long")) if pd.notna(row.get("signal_long")) else False

        ex, ts = _parse_broker_symbol(broker_symbol)
        if not broker_symbol or not ex or not ts:
            plans.append(
                OrderPlan(broker_symbol, ex, ts, 0, suggested_qty, "skip", 0, "missing_broker_symbol")
            )
            continue

        # Read current position
        current_qty = 0
        if kb is not None:
            try:
                current_qty = int(kb.get_position_qty(ts))
            except Exception:
                current_qty = 0

        target_qty = suggested_qty if signal_long else 0
        delta = target_qty - current_qty

        if delta == 0:
            plans.append(OrderPlan(broker_symbol, ex, ts, current_qty, target_qty, "hold", 0, "aligned"))
            continue

        # Clip per-order quantity if set
        order_qty = delta
        if max_per_order_qty is not None and abs(order_qty) > max_per_order_qty:
            order_qty = max(-max_per_order_qty, min(max_per_order_qty, order_qty))

        if dry_run:
            action = "buy" if order_qty > 0 else "sell"
            plans.append(OrderPlan(broker_symbol, ex, ts, current_qty, target_qty, action, int(order_qty), "dry_run"))
            continue

        # Place order
        try:
            if order_qty > 0:
                res = kb.submit_market_order(ex, ts, int(order_qty), side="buy")
                msg = res.message
                action = "buy" if res.submitted else "skip"
            else:
                res = kb.submit_market_order(ex, ts, int(abs(order_qty)), side="sell")
                msg = res.message
                action = "sell" if res.submitted else "skip"
        except Exception as e:
            action = "skip"
            msg = str(e)

        plans.append(OrderPlan(broker_symbol, ex, ts, current_qty, target_qty, action, int(order_qty), msg))

    submitted = sum(1 for p in plans if p.action in {"buy", "sell"})
    skipped = sum(1 for p in plans if p.action in {"skip"})
    summary = RebalanceSummary(planned=len(plans), submitted=submitted, skipped=skipped)
    return summary, plans



