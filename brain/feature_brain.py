"""Deterministic indicator-combination brain used for feature backtests.

The real DeepSeek brain reads all references as text; this brain turns each
learned indicator group into a boolean vote so groups can be enabled/disabled
in historical backtests. It is used only when FEATURE_MASK is set.
"""

from __future__ import annotations

import os
from typing import Any

from config import DEFAULT_RR


def _safe_latest(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    latest = value.get("latest")
    return latest if isinstance(latest, dict) else {}


def _direction(value: str) -> str:
    return str(value or "").upper()


class FeatureBrain:
    """Feature votes over a context dict."""

    def __init__(self, mask: str = "") -> None:
        raw = mask or ""
        self.groups = {
            group.strip()
            for group in raw.split(",")
            if group.strip()
        }
        self.min_optional = 1 if self.groups - {"base"} else 0
        if os.environ.get("FEATURE_MIN_OPTIONAL"):
            self.min_optional = int(os.environ["FEATURE_MIN_OPTIONAL"])
        self.require_base = os.environ.get("FEATURE_REQUIRE_BASE", "0") == "1"
        self.stop_atr = float(os.environ.get("FEATURE_STOP_ATR", "1.5"))
        self.rr = float(os.environ.get("FEATURE_RR", str(DEFAULT_RR)))

    def _vote(self, ctx: dict, direction: str) -> dict:
        direction = _direction(direction)
        votes: dict[str, bool] = {}

        if "base" in self.groups:
            market_state = str(ctx.get("market_state") or "")
            trend = str(ctx.get("trend") or "")
            momentum_ok = bool(ctx.get("momentum_ok"))
            range_signal = _direction(ctx.get("range_signal"))
            if market_state in ("趋势", "trend") and momentum_ok:
                votes["base"] = trend.lower() == direction.lower()
            elif market_state in ("震荡", "range"):
                votes["base"] = range_signal == direction

        if "structure" in self.groups:
            structure = ctx.get("structure") or {}
            structure_trend = str(structure.get("trend") or "").lower()
            health = structure.get("health")
            health_ok = health is None or float(health or 0) >= 35
            votes["structure"] = (
                structure_trend == direction.lower() and health_ok
            )

        if "rsi" in self.groups:
            latest = _safe_latest(ctx.get("momentum"))
            votes["rsi"] = (
                _direction(latest.get("direction")) == direction
                and int(latest.get("confirm_age") or 999) <= 20
            )

        if "cci" in self.groups:
            latest = _safe_latest(ctx.get("cci"))
            votes["cci"] = (
                _direction(latest.get("direction")) == direction
                and int(latest.get("confirm_age") or 999) <= 20
            )

        if "macd" in self.groups:
            macd = ctx.get("macd") or {}
            state = str(macd.get("state") or "")
            latest = macd.get("latest_trigger") if isinstance(
                macd.get("latest_trigger"), dict
            ) else {}
            votes["macd"] = (
                ("bullish" in state and direction == "BUY")
                or ("bearish" in state and direction == "SELL")
                or _direction(latest.get("direction")) == direction
            )

        if "ict" in self.groups:
            ict = ctx.get("ict") or {}
            fvg = ict.get("fvg")
            votes["ict"] = bool(
                isinstance(fvg, dict)
                and _direction(fvg.get("direction")) == direction
                and int(fvg.get("age") or 999) <= 5
            )

        if "key" in self.groups:
            levels = ctx.get("key_levels") or {}
            level = levels.get(
                "support" if direction == "BUY" else "resistance"
            )
            votes["key"] = bool(
                isinstance(level, dict)
                and int(level.get("tests") or 0) >= 1
                and int(level.get("holds") or 0) >= 1
                and float(level.get("hold_rate") or 0) >= 50.0
                and abs(float(level.get("distance_atr") or 999)) <= 3.0
            )
        return votes

    def vote_counts(self, ctx: dict, direction: str) -> int:
        return sum(1 for ok in self._vote(ctx, direction).values() if ok)

    def decide(self, ctx: dict) -> dict:
        positions = ctx.get("positions") or []
        close = float(ctx.get("close", 100))
        atr = max(float(ctx.get("atr", 1)), 0.01)

        buy_votes = self._vote(ctx, "BUY")
        sell_votes = self._vote(ctx, "SELL")
        optional_buy = sum(
            1 for key, ok in buy_votes.items() if ok and key != "base"
        )
        optional_sell = sum(
            1 for key, ok in sell_votes.items() if ok and key != "base"
        )
        base_buy = bool(buy_votes.get("base"))
        base_sell = bool(sell_votes.get("base"))

        action = "WAIT"
        confidence = 45
        reason = "特征投票不足，无明显优势"
        add_on = False

        if base_buy and optional_buy >= self.min_optional:
            action = "BUY"
            confidence = 72
            reason = "主趋势+特征组合确认做多"
        elif base_sell and optional_sell >= self.min_optional:
            action = "SELL"
            confidence = 72
            reason = "主趋势+特征组合确认做空"
        elif (
            not base_buy
            and not base_sell
            and self.min_optional > 0
            and not self.require_base
        ):
            # Range/neutral: optional groups can establish a trade only when
            # at least two different groups point the same way.
            if optional_buy >= max(2, self.min_optional):
                action = "BUY"
                confidence = 70
                reason = "多特征做多投票"
            elif optional_sell >= max(2, self.min_optional):
                action = "SELL"
                confidence = 70
                reason = "多特征做空投票"

        sl_dist = max(atr * self.stop_atr, close * 0.001)
        sl = None
        tp = None
        if action == "BUY":
            sl = round(close - sl_dist, 2)
            tp = round(close + sl_dist * self.rr, 2)
        elif action == "SELL":
            sl = round(close + sl_dist, 2)
            tp = round(close - sl_dist * self.rr, 2)

        close_hint = ctx.get("close_signal")
        if close_hint and positions:
            return {
                "action": "CLOSE",
                "confidence": int(close_hint.get("confidence") or 78),
                "reason": str(close_hint.get("reason") or "持仓反向确认，主动平仓"),
                "sl": None,
                "tp": None,
                "risk_percent": 0.01,
                "add_on": False,
            }

        if action != "WAIT" and positions:
            same_side = any(
                str(pos.get("side", "")).upper() == action for pos in positions
            )
            market_state = str(ctx.get("market_state") or "")
            if (
                same_side
                and market_state in ("趋势", "trend")
                and len(positions) < 3
            ):
                add_on = True
                confidence = max(confidence, 80)
                reason += "；强趋势同向，允许加仓"
            else:
                action = "WAIT"
                confidence = 35
                reason = "已有持仓且未达到加仓条件，等待平仓后重新评估"
                sl = None
                tp = None

        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "sl": sl,
            "tp": tp,
            "risk_percent": 0.01,
            "add_on": add_on,
        }

    def ask_json(self, system: str, user: str) -> dict:
        from brain.llm import extract_context

        return self.decide(extract_context(user))


def feature_decision(mask: str, system: str, user: str) -> dict:
    from brain.llm import extract_context

    return FeatureBrain(mask).decide(extract_context(user))
