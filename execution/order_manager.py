"""V2 order manager: turn a raw AI decision into a safe, sized order."""

from __future__ import annotations

from config import (
    ATR_STOP_MULTIPLIER,
    DEFAULT_RR,
    MAX_RISK_PERCENT,
    MT5_MAGIC,
    SYMBOL,
)
from tools.calculator import (
    default_stop,
    default_target,
    lot_size,
    price_precision,
    quote_to_usd_factor,
    symbol_spec,
)


class OrderManager:
    def build(self, decision: dict, market: dict, account: dict) -> dict | None:
        action = decision.get("action", "WAIT")
        if action == "WAIT":
            return None
        symbol = str(market.get("symbol") or SYMBOL).upper()
        spec = symbol_spec(symbol)
        if not spec:
            raise ValueError(f"unsupported symbol for position sizing: {symbol}")
        digits = price_precision(symbol)
        if action == "CLOSE":
            return {
                "action": "CLOSE",
                "side": "CLOSE",
                "symbol": symbol,
                "entry": round(float(market["close"]), digits),
                "confidence": int(decision.get("confidence", 0)),
                "reason": decision.get("reason", ""),
                "magic": MT5_MAGIC,
            }

        close = float(market["close"])
        atr = float(market.get("atr", close * 0.003))
        risk = float(decision.get("risk_percent") or MAX_RISK_PERCENT)

        sl = decision.get("sl")
        if not sl:
            distance = default_stop(close, atr, ATR_STOP_MULTIPLIER)
            sl = close - distance if action == "BUY" else close + distance

        stop_distance = abs(close - float(sl))
        if stop_distance < close * 0.0002:
            stop_distance = default_stop(close, atr, ATR_STOP_MULTIPLIER)
            sl = close - stop_distance if action == "BUY" else close + stop_distance

        tp = decision.get("tp")
        if not tp:
            tp = (
                default_target(close, stop_distance, DEFAULT_RR)
                if action == "BUY"
                else close - stop_distance * DEFAULT_RR
            )

        volume = lot_size(
            account["balance"],
            risk,
            stop_distance,
            contract_size=float(spec["contract_size"]),
            quote_to_usd=quote_to_usd_factor(symbol, close),
        )
        return {
            "action": action,
            "side": action,
            "symbol": symbol,
            "entry": round(close, digits),
            "sl": round(float(sl), digits),
            "tp": round(float(tp), digits),
            "volume": volume,
            "risk_percent": round(risk, 4),
            "confidence": int(decision.get("confidence", 0)),
            "reason": decision.get("reason", ""),
            "magic": MT5_MAGIC,
        }
