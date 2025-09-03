from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kiteconnect import KiteConnect

from .settings import get_kite_settings


@dataclass
class OrderResult:
    submitted: bool
    id: Optional[str]
    message: str


class KiteBroker:
    def __init__(self) -> None:
        s = get_kite_settings()
        self.kite = KiteConnect(api_key=s.api_key)
        self.kite.set_access_token(s.access_token)

    def get_last_price(self, symbol: str) -> float:
        # symbol example: 'NSE:RELIANCE'
        q = self.kite.ltp([symbol])
        data = q.get(symbol)
        return float(data["last_price"]) if data else 0.0

    def get_net_equity(self) -> float:
        # Approximate with available cash + holdings value
        p = self.kite.margins(segment="equity")
        return float(p.get("net", 0.0))

    def get_position_qty(self, trading_symbol: str) -> float:
        pos = self.kite.positions()
        net = pos.get("net", [])
        for p in net:
            if p.get("tradingsymbol") == trading_symbol:
                return float(p.get("quantity", 0))
        return 0.0

    def submit_market_order(self, exchange: str, tradingsymbol: str, qty: int, side: str) -> OrderResult:
        try:
            tx = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if side == "buy" else self.kite.TRANSACTION_TYPE_SELL,
                quantity=int(qty),
                product=self.kite.PRODUCT_CNC,
                order_type=self.kite.ORDER_TYPE_MARKET,
                validity=self.kite.VALIDITY_DAY,
            )
            return OrderResult(submitted=True, id=str(tx.get("order_id")), message="ok")
        except Exception as e:
            return OrderResult(submitted=False, id=None, message=str(e))

    def close_position(self, exchange: str, tradingsymbol: str) -> OrderResult:
        try:
            qty = self.get_position_qty(tradingsymbol)
            if qty > 0:
                tx = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=tradingsymbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=int(qty),
                    product=self.kite.PRODUCT_CNC,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_DAY,
                )
                return OrderResult(submitted=True, id=str(tx.get("order_id")), message="ok")
            return OrderResult(submitted=True, id=None, message="no position")
        except Exception as e:
            return OrderResult(submitted=False, id=None, message=str(e))
