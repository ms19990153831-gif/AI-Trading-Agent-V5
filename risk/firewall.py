"""Trading safety firewall: hard limits that an AI cannot bypass."""

from __future__ import annotations

from config import (
    CLOSE_MIN_CONFIDENCE,
    DAILY_LOSS_LIMIT,
    ENABLE_DAILY_LOSS_LIMIT,
    ENABLE_LOSS_PAUSE,
    INITIAL_BALANCE,
    MAX_OPEN_POSITIONS,
    MAX_RISK_PERCENT,
    MIN_CONFIDENCE,
    RANGE_MIN_CONFIDENCE,
    TREND_MIN_CONFIDENCE,
)
from tools.structure import boundary_guard


class Firewall:
    def check(self, decision: dict, account: dict, market: dict | None = None) -> dict:
        if decision.get("action") == "WAIT":
            return {"allow": False, "code": "WAIT", "reasons": ["no trade requested"]}

        reasons: list[str] = []
        if decision.get("action") == "CLOSE":
            positions = account.get("positions") or []
            confidence = float(decision.get("confidence", 0))
            if not positions:
                reasons.append("CLOSE 但当前无持仓")
            if confidence < float(CLOSE_MIN_CONFIDENCE):
                reasons.append(
                    f"CLOSE confidence {confidence:.0f} below "
                    f"{float(CLOSE_MIN_CONFIDENCE):.0f}"
                )
            return {
                "allow": not reasons,
                "code": "ALLOW" if not reasons else "BLOCKED",
                "reasons": reasons,
            }

        if market:
            guard = boundary_guard(
                decision.get("action"),
                market.get("indicators", {}),
                market.get("vision"),
            )
            if guard:
                reasons.append(str(guard["reason"]))
        confidence = float(decision.get("confidence", 0))
        risk = float(decision.get("risk_percent") or MAX_RISK_PERCENT)
        threshold = MIN_CONFIDENCE
        if decision.get("market_state") == "趋势":
            threshold = TREND_MIN_CONFIDENCE
        elif decision.get("market_state") == "震荡":
            threshold = RANGE_MIN_CONFIDENCE

        if confidence < threshold:
            reasons.append(f"confidence {confidence:.0f} below {threshold}")
        if risk > MAX_RISK_PERCENT:
            reasons.append(f"risk {risk:.2%} above {MAX_RISK_PERCENT:.2%}")
        if account["open_positions"] >= MAX_OPEN_POSITIONS:
            reasons.append("max open positions reached")
        if (
            ENABLE_DAILY_LOSS_LIMIT
            and account["daily_pnl"]
            <= -(float(account.get("balance") or INITIAL_BALANCE))
            * DAILY_LOSS_LIMIT
        ):
            reasons.append("daily loss limit reached")
        if ENABLE_LOSS_PAUSE and account.get("loss_paused"):
            reasons.append("连亏暂停交易（24小时）")

        return {
            "allow": not reasons,
            "code": "ALLOW" if not reasons else "BLOCKED",
            "reasons": reasons,
        }
