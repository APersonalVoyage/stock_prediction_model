from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import LatestTradeRequest

from .settings import get_alpaca_settings


@dataclass
class OrderResult:
    submitted: bool
    id: Optional[str]
    message: str


class AlpacaBroker:
    def __init__(self) -> None:
        s = get_alpaca_settings()
        self.trading = TradingClient(api_key=s.api_key, secret_key=s.secret_key, paper=s.paper)
        self.data = StockHistoricalDataClient(api_key=s.api_key, secret_key=s.secret_key)

    def get_account(self):
        return self.trading.get_account()

    def get_position_qty(self, symbol: str) -> float:
        try:
            pos = self.trading.get_open_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

    def get_last_price(self, symbol: str) -> float:
        req = LatestTradeRequest(symbol_or_symbols=symbol)
        trade = self.data.get_latest_trade(req)
        return float(trade.price)

    def submit_market_order(self, symbol: str, qty: float, side: str) -> OrderResult:
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            o = self.trading.submit_order(order)
            return OrderResult(submitted=True, id=str(o.id), message="ok")
        except Exception as e:
            return OrderResult(submitted=False, id=None, message=str(e))

    def close_position(self, symbol: str) -> OrderResult:
        try:
            o = self.trading.close_position(symbol)
            return OrderResult(submitted=True, id=str(o.id) if hasattr(o, 'id') else None, message="ok")
        except Exception as e:
            return OrderResult(submitted=False, id=None, message=str(e))

    def cancel_all(self) -> None:
        try:
            self.trading.cancel_orders()
        except Exception:
            pass
