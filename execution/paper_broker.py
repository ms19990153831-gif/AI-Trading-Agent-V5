"""Offline paper broker: fills orders and manages SL/TP, trailing stop,
breakeven and partial take profit across simulated ticks."""

from __future__ import annotations

import json
from datetime import datetime

from config import (
    BREAKEVEN_AT_RR,
    COMMISSION_PER_LOT,
    PARTIAL_TP_RATIO,
    TRAILING_ACTIVATE_RR,
    TRAILING_DISTANCE_RR,
)
from tools.calculator import (
    price_precision,
    quote_to_usd_factor,
    symbol_spec,
    symbol_spread,
)


class PaperBroker:
    def __init__(self, db) -> None:
        self.db = db

    def place(self, order: dict, ts: str | None = None) -> dict:
        trade = {
            "ts": ts or datetime.now().isoformat(timespec="seconds"),
            "symbol": order.get("symbol"),
            "side": order["action"],
            "volume": float(order["volume"]),
            "entry": float(order["entry"]),
            "sl": float(order["sl"]),
            "tp": float(order["tp"]),
            "magic": order.get("magic", 0),
            "reason": order.get("reason", ""),
            "confidence": order.get("confidence", 0),
            "market_state": order.get("market_state"),
            "persona": order.get("persona"),
        }
        trade_id = self.db.save_trade(trade)
        trade["id"] = trade_id
        return trade

    def tick(self, price: float) -> list[dict]:
        events: list[dict] = []
        for trade in self.db.open_trades():
            events.extend(self._manage(trade, price))
        return events

    def bar_tick(
        self,
        high: float,
        low: float,
        close: float,
        ts: str | None = None,
    ) -> list[dict]:
        """Bar-level management for backtests: SL/TP checked inside the bar."""
        events: list[dict] = []
        for trade in self.db.open_trades():
            events.extend(self._bar_manage(trade, high, low, close, ts))
        return events

    def close_all(self, price: float, ts: str | None = None) -> list[dict]:
        """Liquidate all open positions at the given price (backtest end)."""
        events: list[dict] = []
        for trade in self.db.open_trades():
            side = trade["side"]
            entry = float(trade["entry"])
            volume = float(trade["volume"])
            notes = {}
            if trade.get("notes"):
                try:
                    notes = json.loads(trade["notes"])
                except json.JSONDecodeError:
                    notes = {}
            closed_volume = float(notes.get("closed_volume", 0.0))
            closed_pnl = float(notes.get("closed_pnl", 0.0))
            remaining = max(volume - closed_volume, 0.01)
            pnl = self._pnl(
                side,
                entry,
                price,
                remaining,
                symbol=trade["symbol"],
            ) + closed_pnl
            self._close(trade, price, remaining, pnl, ts)
            events.append(
                {
                    "type": "CLOSE_END",
                    "trade_id": trade["id"],
                    "price": round(price, 2),
                    "volume": remaining,
                    "pnl": round(pnl, 2),
                }
            )
        return events

    def _bar_manage(
        self,
        trade: dict,
        high: float,
        low: float,
        close: float,
        ts: str | None = None,
    ) -> list[dict]:
        side = trade["side"]
        entry = float(trade["entry"])
        sl = float(trade["sl_current"] or trade["sl"])
        tp = float(trade["tp"])
        volume = float(trade["volume"])

        notes = {}
        if trade.get("notes"):
            try:
                notes = json.loads(trade["notes"])
            except json.JSONDecodeError:
                notes = {}
        closed_volume = float(notes.get("closed_volume", 0.0))
        closed_pnl = float(notes.get("closed_pnl", 0.0))
        remaining = max(volume - closed_volume, 0.01)

        if side == "BUY":
            if low <= sl:
                pnl = self._pnl(
                    "BUY", entry, sl, remaining, symbol=trade["symbol"]
                ) + closed_pnl
                self._close(trade, sl, remaining, pnl, ts)
                return [
                    {
                        "type": "CLOSE_SL",
                        "trade_id": trade["id"],
                        "price": round(sl, 2),
                        "volume": remaining,
                        "pnl": round(pnl, 2),
                    }
                ]
            if high >= tp:
                pnl = self._pnl(
                    "BUY", entry, tp, remaining, symbol=trade["symbol"]
                ) + closed_pnl
                self._close(trade, tp, remaining, pnl, ts)
                return [
                    {
                        "type": "CLOSE_TP",
                        "trade_id": trade["id"],
                        "price": round(tp, 2),
                        "volume": remaining,
                        "pnl": round(pnl, 2),
                    }
                ]
        else:
            if high >= sl:
                pnl = self._pnl(
                    "SELL", entry, sl, remaining, symbol=trade["symbol"]
                ) + closed_pnl
                self._close(trade, sl, remaining, pnl, ts)
                return [
                    {
                        "type": "CLOSE_SL",
                        "trade_id": trade["id"],
                        "price": round(sl, 2),
                        "volume": remaining,
                        "pnl": round(pnl, 2),
                    }
                ]
            if low <= tp:
                pnl = self._pnl(
                    "SELL", entry, tp, remaining, symbol=trade["symbol"]
                ) + closed_pnl
                self._close(trade, tp, remaining, pnl, ts)
                return [
                    {
                        "type": "CLOSE_TP",
                        "trade_id": trade["id"],
                        "price": round(tp, 2),
                        "volume": remaining,
                        "pnl": round(pnl, 2),
                    }
                ]
        return self._manage(trade, close, ts)

    def _manage(self, trade: dict, price: float, ts: str | None = None) -> list[dict]:
        side = trade["side"]
        entry = float(trade["entry"])
        sl = float(trade["sl_current"] or trade["sl"])
        tp = float(trade["tp"])
        volume = float(trade["volume"])
        events: list[dict] = []

        notes = {}
        if trade.get("notes"):
            try:
                notes = json.loads(trade["notes"])
            except json.JSONDecodeError:
                notes = {}
        closed_volume = float(notes.get("closed_volume", 0.0))
        closed_pnl = float(notes.get("closed_pnl", 0.0))
        remaining = max(volume - closed_volume, 0.01)

        risk_distance = abs(entry - float(trade["sl"]))
        if risk_distance <= 0:
            risk_distance = max(abs(entry - tp), 0.01)
        profit = (price - entry) if side == "BUY" else (entry - price)

        # Partial take profit on half of the position.
        partial_price = (
            entry + (tp - entry) * 0.5 if side == "BUY" else entry - (entry - tp) * 0.5
        )
        partial_leaves_minimum = (
            round(volume - round(volume * PARTIAL_TP_RATIO, 2), 2) >= 0.01
        )
        if not trade["partial_done"] and partial_leaves_minimum and (
            (side == "BUY" and price >= partial_price)
            or (side == "SELL" and price <= partial_price)
        ):
            close_volume = round(volume * PARTIAL_TP_RATIO, 2)
            pnl = self._pnl(
                side, entry, price, close_volume, symbol=trade["symbol"]
            )
            notes["closed_volume"] = close_volume
            notes["closed_pnl"] = pnl
            self.db.update_trade(
                trade["id"],
                partial_done=1,
                notes=json.dumps(notes),
            )
            events.append(
                {
                    "type": "PARTIAL_TP",
                    "trade_id": trade["id"],
                    "price": round(price, 2),
                    "volume": close_volume,
                    "pnl": round(pnl, 2),
                }
            )

        # Stop loss.
        if (side == "BUY" and price <= sl) or (side == "SELL" and price >= sl):
            pnl = self._pnl(side, entry, sl, remaining, symbol=trade["symbol"])
            total = pnl + closed_pnl
            self._close(trade, sl, remaining, total, ts)
            events.append(
                {
                    "type": "CLOSE_SL",
                    "trade_id": trade["id"],
                    "price": round(sl, 2),
                    "volume": remaining,
                    "pnl": round(total, 2),
                }
            )
            return events

        # Take profit.
        if (side == "BUY" and price >= tp) or (side == "SELL" and price <= tp):
            pnl = self._pnl(side, entry, tp, remaining, symbol=trade["symbol"])
            total = pnl + closed_pnl
            self._close(trade, tp, remaining, total, ts)
            events.append(
                {
                    "type": "CLOSE_TP",
                    "trade_id": trade["id"],
                    "price": round(tp, 2),
                    "volume": remaining,
                    "pnl": round(total, 2),
                }
            )
            return events

        # Breakeven: move stop to entry once the trade is a little ahead.
        breakeven_price = entry + risk_distance * BREAKEVEN_AT_RR if side == "BUY" else entry - risk_distance * BREAKEVEN_AT_RR
        if (
            (side == "BUY" and price >= breakeven_price and sl < entry)
            or (side == "SELL" and price <= breakeven_price and sl > entry)
        ):
            self.db.update_trade(trade["id"], sl_current=entry)
            events.append(
                {
                    "type": "BREAKEVEN",
                    "trade_id": trade["id"],
                    "price": round(price, 2),
                    "sl": round(entry, 2),
                }
            )
            sl = entry

        # Trailing stop once the trade reaches a solid profit.
        activate_price = entry + risk_distance * TRAILING_ACTIVATE_RR if side == "BUY" else entry - risk_distance * TRAILING_ACTIVATE_RR
        trail_distance = risk_distance * TRAILING_DISTANCE_RR
        new_sl = (
            price - trail_distance if side == "BUY" else price + trail_distance
        )
        if (
            (side == "BUY" and price >= activate_price and new_sl > sl)
            or (side == "SELL" and price <= activate_price and new_sl < sl)
        ):
            self.db.update_trade(trade["id"], sl_current=round(new_sl, 2))
            events.append(
                {
                    "type": "TRAILING",
                    "trade_id": trade["id"],
                    "price": round(price, 2),
                    "sl": round(new_sl, 2),
                }
            )

        return events

    def _close(
        self,
        trade: dict,
        price: float,
        volume: float,
        pnl: float,
        ts: str | None = None,
    ) -> None:
        self.db.update_trade(
            trade["id"],
            status="closed",
            exit_price=round(price, price_precision(trade.get("symbol"))),
            exit_ts=ts or datetime.now().isoformat(timespec="seconds"),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl / max(volume * float(trade["entry"]) * 100, 1.0) * 100, 4),
        )

    @staticmethod
    def _pnl(
        side: str,
        entry: float,
        exit_price: float,
        volume: float,
        symbol: str | None = None,
    ) -> float:
        direction = 1.0 if side == "BUY" else -1.0
        spec = symbol_spec(symbol) or {
            "contract_size": 100,
            "quote_currency": "USD",
        }
        conversion = quote_to_usd_factor(
            symbol or "XAUUSD",
            (entry + exit_price) / 2.0,
        )
        contract = float(spec["contract_size"])
        gross = (
            (exit_price - entry)
            * direction
            * volume
            * contract
            * conversion
        )
        costs = symbol_spread(symbol) * volume * contract * conversion
        costs += float(COMMISSION_PER_LOT) * volume
        return round(gross - costs, 2)
