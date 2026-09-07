"""Risk manager: firewall + order building for a raw AI decision."""

from __future__ import annotations

from config import (
    ADDON_MIN_ADX,
    ADDON_MIN_CONFIDENCE,
    DAILY_LOSS_LIMIT,
    ENABLE_MIN_LOT_RISK_GUARD,
    ENABLE_DAILY_LOSS_LIMIT,
    ENABLE_LOSS_PAUSE,
    ENABLE_SIDE_LOSS_PAUSE,
    INITIAL_BALANCE,
    MAX_OPEN_POSITIONS,
    MAX_RISK_PERCENT,
    MIN_LOT_RISK_TOLERANCE,
    SIDE_LOSS_PAUSE_HOURS,
    SIDE_LOSS_PAUSE_STREAK,
)
from execution.order_manager import OrderManager
from datetime import datetime
from mt5.account import Account
from risk.firewall import Firewall
from tools.calculator import (
    quote_to_usd_factor,
    stop_loss_risk_percent,
    symbol_spec,
)


class RiskManager:
    def __init__(self, db, account=None) -> None:
        self.db = db
        self.firewall = Firewall()
        self.order_manager = OrderManager()
        self.account = account or Account(db)

    def review(
        self,
        decision: dict,
        market: dict,
        positions: list | None = None,
    ) -> dict:
        account = self.account.snapshot()
        if positions is None:
            positions = account.get("positions") or []
        decision, policy_reason = self._apply_addon_policy(
            decision, market, positions
        )
        side_pause_reason = self._side_loss_pause_reason(
            str(decision.get("action", "")).upper()
        )
        if side_pause_reason:
            decision = {
                **decision,
                "action": "WAIT",
                "add_on": False,
                "confidence": min(float(decision.get("confidence") or 0), 40),
                "reason": side_pause_reason,
            }
            policy_reason = side_pause_reason
        check = self.firewall.check(decision, account, market)
        reasons = list(check["reasons"])
        if policy_reason:
            reasons = [
                policy_reason
            ] + [reason for reason in reasons if reason != "no trade requested"]
            check = {**check, "reasons": reasons}
        order = (
            self.order_manager.build(decision, market, account)
            if check["allow"]
            else None
        )
        if (
            order is not None
            and order["action"] in ("BUY", "SELL")
            and ENABLE_MIN_LOT_RISK_GUARD
        ):
            guard_reason = self._stop_loss_risk_guard(order, account)
            if guard_reason:
                reasons.append(guard_reason)
                check = {
                    "allow": False,
                    "code": "BLOCKED",
                    "reasons": reasons,
                }
                order = None
        return {
            "allow": check["allow"],
            "code": check["code"],
            "reasons": check["reasons"],
            "order": order,
            "account": account,
        }

    def hard_pause_reason(self, account: dict | None = None) -> str | None:
        """Return a reason when hard risk locks the day before any AI call."""
        account = account or self.account.snapshot()
        balance = float(account.get("balance") or INITIAL_BALANCE)
        if (
            ENABLE_DAILY_LOSS_LIMIT
            and float(account.get("daily_pnl") or 0.0)
            <= -balance * DAILY_LOSS_LIMIT
        ):
            return "daily loss limit reached"
        if ENABLE_LOSS_PAUSE and account.get("loss_paused"):
            return "连亏暂停交易（24小时）"
        return None

    @staticmethod
    def _apply_addon_policy(
        decision: dict,
        market: dict,
        positions: list[dict],
    ) -> tuple[dict, str | None]:
        """Block repeated entries; only strong-trend AI add-ons are allowed."""
        action = decision.get("action")
        if action not in ("BUY", "SELL") or not positions:
            return decision, None
        symbol = str((market or {}).get("symbol") or "").upper()
        same_symbol = (not symbol) or any(
            str(pos.get("symbol") or "").upper() == symbol
            for pos in positions
        )
        if not same_symbol:
            if len(positions) < MAX_OPEN_POSITIONS:
                return decision, None
            reason = (
                f"已达最大持仓数{int(MAX_OPEN_POSITIONS)}，"
                "等待其他品种平仓后再开新仓"
            )
            return (
                {
                    **decision,
                    "action": "WAIT",
                    "add_on": False,
                    "confidence": min(float(decision.get("confidence") or 0), 40),
                    "reason": reason,
                },
                reason,
            )
        same_side = any(
            str(pos.get("side", "")).upper() == action for pos in positions
        )
        indicators = market.get("indicators", {})
        strong_trend = (
            indicators.get("market_state") == "trend"
            or decision.get("market_state") == "趋势"
        )
        adx = float(indicators.get("adx") or 0)
        add_on_ok = (
            same_side
            and strong_trend
            and bool(decision.get("add_on"))
            and float(decision.get("confidence") or 0) >= ADDON_MIN_CONFIDENCE
            and adx >= ADDON_MIN_ADX
            and len(positions) < MAX_OPEN_POSITIONS
        )
        if not add_on_ok:
            reason = (
                "已有持仓且未达到加仓条件"
                "（需要AI加仓信号+强趋势+高置信度），等待趋势强化或平仓"
            )
            return (
                {
                    **decision,
                    "action": "WAIT",
                    "add_on": False,
                    "confidence": min(float(decision.get("confidence") or 0), 40),
                    "reason": reason,
                },
                reason,
            )
        return decision, None

    def _side_loss_pause_reason(self, action: str) -> str | None:
        """Pause the same side after consecutive stop-style losses."""
        if not ENABLE_SIDE_LOSS_PAUSE or self.db is None:
            return None
        if action not in ("BUY", "SELL"):
            return None
        rows = self.db.closed_trades(limit=200)
        same_side = [
            row
            for row in rows
            if str(row.get("side", "")).upper() == action
            and row.get("exit_ts")
            and row.get("pnl") is not None
        ]
        same_side.sort(
            key=lambda row: str(row.get("exit_ts") or ""),
            reverse=True,
        )
        loss_streak = 0
        last_loss_ts = None
        for row in same_side:
            pnl = float(row.get("pnl") or 0.0)
            if pnl < 0:
                loss_streak += 1
                last_loss_ts = row.get("exit_ts")
                if loss_streak >= int(SIDE_LOSS_PAUSE_STREAK):
                    break
            else:
                break
        if (
            loss_streak < int(SIDE_LOSS_PAUSE_STREAK)
            or last_loss_ts is None
        ):
            return None
        try:
            last = datetime.fromisoformat(
                str(last_loss_ts).replace("Z", "+00:00")
            )
            now = (
                datetime.now().astimezone()
                if last.tzinfo is not None
                else datetime.now()
            )
            if last.tzinfo is not None and now.tzinfo is None:
                now = now.astimezone()
            if (now - last).total_seconds() >= float(SIDE_LOSS_PAUSE_HOURS) * 3600:
                return None
        except ValueError:
            return None
        return (
            f"同方向连续{int(SIDE_LOSS_PAUSE_STREAK)}笔亏损，"
            f"暂停{action}新开仓{float(SIDE_LOSS_PAUSE_HOURS):.0f}小时"
        )

    @staticmethod
    def _stop_loss_risk_guard(order: dict, account: dict) -> str | None:
        """Reject entries whose real stop-loss risk exceeds the configured cap.

        Order sizing clamps to the minimum lot, so a small account can end up
        risking far more than MAX_RISK_PERCENT on a 0.01 lot XAUUSD order.
        """
        balance = float(account.get("balance") or 0.0)
        entry = float(order.get("entry") or 0.0)
        sl = float(order.get("sl") or 0.0)
        volume = float(order.get("volume") or 0.0)
        symbol = str(order.get("symbol") or "").upper()
        if balance <= 0 or entry <= 0 or sl <= 0 or volume <= 0:
            return None
        stop_distance = abs(entry - sl)
        spec = symbol_spec(symbol)
        if not spec:
            return f"不支持品种合约参数，拒绝开仓：{symbol or 'unknown'}"
        actual = stop_loss_risk_percent(
            balance,
            volume,
            stop_distance,
            contract_size=float(spec["contract_size"]),
            quote_to_usd=quote_to_usd_factor(symbol, entry),
        )
        limit = MAX_RISK_PERCENT * (1.0 + MIN_LOT_RISK_TOLERANCE)
        if actual <= limit:
            return None
        return (
            f"账户余额不足：{volume:.2f}手止损距离{stop_distance:.2f}的"
            f"潜在风险约{actual:.1%}，超过单笔上限约{limit:.1%}；"
            "小账户被0.01手最小仓放大风险，本次拒绝开仓"
        )
