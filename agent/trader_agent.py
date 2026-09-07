"""Trader agent: the LLM decision maker."""

from __future__ import annotations

import json

from brain.feature_brain import FeatureBrain
from brain.llm import get_llm
from brain.prompt import TRADER_SYSTEM
from config import (
    ENABLE_RULE_VETO,
    ENABLE_AI_CLOSE,
    ENABLE_REVERSAL_SIGNAL,
    ENABLE_STRUCTURE_SIGNAL,
    MAX_RISK_PERCENT,
    RULE_VETO_GAP,
    RULE_VETO_MASK,
)
from tools.calculator import normalize_risk
from tools.market_structure import structure_reference_text
from tools.quality import entry_quality
from tools.structure import (
    boundary_guard,
    close_signal,
    extreme_chase_guard,
    reversal_signal,
    structure_signal,
    swing_reversal_signal,
    trend_conflict_guard,
    trend_exhaustion_guard,
    vision_reversal_guard,
)


class TraderAgent:
    def __init__(self, llm=None) -> None:
        self.llm = llm or get_llm()

    def _rule_veto(
        self,
        context: dict,
        action: str,
        indicators: dict,
    ) -> dict | None:
        """Veto an AI trade when deterministic feature votes strongly oppose it."""
        if not ENABLE_RULE_VETO:
            return None
        action = str(action or "").upper()
        if action not in ("BUY", "SELL"):
            return None
        brain = FeatureBrain(RULE_VETO_MASK)
        buy_votes = brain.vote_counts(context, "BUY")
        sell_votes = brain.vote_counts(context, "SELL")
        gap = buy_votes - sell_votes if action == "SELL" else sell_votes - buy_votes
        if gap < int(RULE_VETO_GAP):
            return None
        two_b_bottom = bool(indicators.get("two_b_bottom"))
        two_b_top = bool(indicators.get("two_b_top"))
        # A confirmed 2B/false-breakout reclaim is the exception the AI can cite.
        if (action == "SELL" and two_b_top) or (
            action == "BUY" and two_b_bottom
        ):
            return None
        return {
            "action": action,
            "block": True,
            "reason": (
                f"规则复核否决：确定性特征票与AI方向相反，"
                f"差额{gap}≥{RULE_VETO_GAP}，"
                "且无2B/假突破收回确认，AI可自主等待但不能强行开仓"
            ),
        }

    def decide(
        self,
        market: dict,
        vision: dict | None = None,
        macro: dict | None = None,
        memory: list | None = None,
        strategy: dict | None = None,
        positions: list | None = None,
    ) -> dict:
        indicators = market["indicators"]
        strategy_advisor = strategy or {}
        position_list = positions or []
        vision_view = (vision or {}).get("vision", {})
        structure_ref = indicators.get("structure") or {}
        structure = (
            structure_signal(indicators, vision_view)
            if ENABLE_STRUCTURE_SIGNAL
            else None
        )
        if ENABLE_REVERSAL_SIGNAL:
            reversal = swing_reversal_signal(
                market.get("df"), indicators
            )
            if reversal is None and market.get("df") is None:
                reversal = reversal_signal(indicators, vision_view)
        else:
            reversal = None
        close_hint = close_signal(indicators, position_list, vision_view)
        show_2b = ENABLE_REVERSAL_SIGNAL or ENABLE_AI_CLOSE
        context = {
            "symbol": market["symbol"],
            "timeframe": market["timeframe"],
            "close": indicators["close"],
            "rsi": indicators["rsi"],
            "atr": indicators["atr"],
            "trend": indicators["trend"],
            "htf_trend": indicators.get("htf_trend"),
            "market_state": indicators.get("market_state"),
            "momentum_ok": indicators.get("momentum_ok"),
            "range_signal": indicators.get("range_signal"),
            "structure_signal": structure,
            "reversal_signal": reversal,
            "close_signal": close_hint,
            "donchian_high": indicators.get("donchian_high"),
            "donchian_low": indicators.get("donchian_low"),
            "box_zone_top": indicators.get("box_zone_top"),
            "box_zone_bottom": indicators.get("box_zone_bottom"),
            "two_b_bottom": (
                indicators.get("two_b_bottom") if show_2b else None
            ),
            "two_b_top": indicators.get("two_b_top") if show_2b else None,
            "structure": structure_ref,
            "momentum": indicators.get("momentum") or {},
            "momentum_reference": indicators.get("momentum_reference"),
            "key_levels": indicators.get("key_levels") or {},
            "key_level_reference": indicators.get("key_level_reference"),
            "cci": indicators.get("cci") or {},
            "cci_reference": indicators.get("cci_reference"),
            "ict": indicators.get("ict") or {},
            "ict_reference": indicators.get("ict_reference"),
            "macd": indicators.get("macd") or {},
            "macd_reference": indicators.get("macd_reference"),
            "trend_phase": indicators.get("trend_phase") or {},
            "trend_phase_reference": indicators.get("trend_phase_reference"),
            "vision": vision_view,
            "macro": macro or {},
            "memory": memory or [],
            "positions": position_list,
            "strategy_advisor": strategy_advisor,
        }
        advisor_text = ""
        if strategy_advisor:
            advisor_text = (
                f"\n当前行情状态：{strategy_advisor.get('market_state', '未知')}；"
                f"策略顾问建议优先考虑 {strategy_advisor.get('preferred_persona', '当前逻辑')} "
                f"逻辑（{strategy_advisor.get('logic', '')}）。"
                "这只是参考倾向，不是强制规则；请结合K线、指标和风险自主判断，"
                "若信号不支持可以等待或反向。"
            )
        user = (
            "CONTEXT_JSON: "
            + json.dumps(context, ensure_ascii=False)
            + "\n结构参考: "
            + structure_reference_text(structure_ref)
            + "\n背离参考: "
            + str(indicators.get("momentum_reference") or "暂无动量背离")
            + "\n关键位参考: "
            + str(indicators.get("key_level_reference") or "暂无关键位")
            + "\nCCI参考: "
            + str(indicators.get("cci_reference") or "暂无CCI背离")
            + "\nICT参考: "
            + str(indicators.get("ict_reference") or "暂无ICT结构")
            + "\nMACD参考: "
            + str(indicators.get("macd_reference") or "暂无MACD状态")
            + "\n阶段参考: "
            + str(indicators.get("trend_phase_reference") or "暂无阶段判断")
            + advisor_text
            + "\nReturn your trading decision as strict JSON."
        )
        try:
            decision = self.llm.ask_json(TRADER_SYSTEM, user)
        except Exception as exc:
            decision = {
                "action": "WAIT",
                "confidence": 0,
                "reason": f"llm unavailable: {type(exc).__name__}",
                "risk_percent": MAX_RISK_PERCENT,
            }
        if "raw" in decision or "action" not in decision:
            decision = {
                "action": "WAIT",
                "confidence": 0,
                "reason": "LLM output was not valid JSON",
                "risk_percent": MAX_RISK_PERCENT,
            }

        action = str(decision.get("action", "WAIT")).upper()
        decision["action"] = (
            action if action in ("BUY", "SELL", "WAIT", "CLOSE") else "WAIT"
        )
        decision["confidence"] = min(
            100, max(0, int(float(decision.get("confidence", 0))))
        )
        decision["risk_percent"] = min(
            normalize_risk(
                float(decision.get("risk_percent") or MAX_RISK_PERCENT)
            ),
            MAX_RISK_PERCENT,
        )
        decision["add_on"] = bool(decision.get("add_on"))
        decision["reason"] = str(decision.get("reason", ""))[:300]
        decision["market_state"] = (
            strategy.get("market_state") if strategy else None
        )

        rule_veto = self._rule_veto(context, decision["action"], indicators)
        if rule_veto and not reversal:
            decision = {
                **decision,
                "action": "WAIT",
                "confidence": min(
                    float(decision.get("confidence") or 0), 40
                ),
                "reason": str(rule_veto["reason"])[:300],
                "add_on": False,
            }

        guard = boundary_guard(decision["action"], indicators, vision_view)
        original_guard = guard
        if reversal:
            decision = {
                **decision,
                "action": reversal["action"],
                "confidence": int(reversal["confidence"]),
                "reason": str(reversal["reason"])[:300],
                "add_on": False,
                "sl": None,
                "tp": None,
            }
            guard = boundary_guard(decision["action"], indicators, vision_view)
        elif decision["action"] == "WAIT" and not position_list and not rule_veto:
            if structure:
                decision = {
                    **decision,
                    "action": structure["action"],
                    "confidence": int(structure["confidence"]),
                    "reason": str(structure["reason"])[:300],
                    "add_on": False,
                    "sl": None,
                    "tp": None,
                    "market_state": decision.get("market_state") or "震荡",
                }

        held_sides = {
            str(pos.get("side", "")).upper() for pos in position_list
        }
        if decision["action"] == "BUY" and "SELL" in held_sides:
            decision = {
                **decision,
                "action": "CLOSE",
                "reason": "AI决策为BUY，与SELL持仓方向相反，先平仓",
                "add_on": False,
            }
        elif decision["action"] == "SELL" and "BUY" in held_sides:
            decision = {
                **decision,
                "action": "CLOSE",
                "reason": "AI决策为SELL，与BUY持仓方向相反，先平仓",
                "add_on": False,
            }
        elif decision["action"] == "WAIT" and position_list and close_hint:
            decision = {
                **decision,
                "action": "CLOSE",
                "confidence": int(close_hint["confidence"]),
                "reason": str(close_hint["reason"])[:300],
                "add_on": False,
            }

        vision_guard = vision_reversal_guard(decision["action"], vision_view)
        if vision_guard:
            guarded_action = str(vision_guard.get("action", "")).upper()
            same_side_held = guarded_action in held_sides
            if position_list and same_side_held:
                decision = {
                    **decision,
                    "action": "CLOSE",
                    "confidence": max(
                        int(float(decision.get("confidence") or 0)), 74
                    ),
                    "reason": str(vision_guard["reason"])[:300]
                    + "；已有同向持仓，主动平仓",
                    "add_on": False,
                }
            elif not position_list:
                decision = {
                    **decision,
                    "action": "WAIT",
                    "confidence": min(
                        float(decision.get("confidence") or 0), 40
                    ),
                    "reason": str(vision_guard["reason"])[:300],
                    "add_on": False,
                }
            else:
                decision = {
                    **decision,
                    "action": "WAIT",
                    "confidence": min(
                        float(decision.get("confidence") or 0), 40
                    ),
                    "reason": str(vision_guard["reason"])[:300],
                    "add_on": False,
                }

        extreme_guard = extreme_chase_guard(
            decision["action"], indicators
        )
        if extreme_guard:
            decision = {
                **decision,
                "action": "WAIT",
                "confidence": min(
                    float(decision.get("confidence") or 0), 40
                ),
                "reason": str(extreme_guard["reason"])[:300],
                "add_on": False,
            }

        exhaustion_guard = trend_exhaustion_guard(
            decision["action"], indicators
        )
        if exhaustion_guard:
            decision = {
                **decision,
                "action": "WAIT",
                "confidence": min(
                    float(decision.get("confidence") or 0), 40
                ),
                "reason": str(exhaustion_guard["reason"])[:300],
                "add_on": False,
            }

        conflict_guard = trend_conflict_guard(
            decision["action"], indicators
        )
        if conflict_guard:
            decision = {
                **decision,
                "action": "WAIT",
                "confidence": min(
                    float(decision.get("confidence") or 0), 40
                ),
                "reason": str(conflict_guard["reason"])[:300],
                "add_on": False,
            }

        strategy_events = []
        if original_guard:
            strategy_events.append(
                {
                    "event_type": "boundary_guard",
                    "action": original_guard.get("action"),
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "box_low": indicators.get("donchian_low"),
                    "box_high": indicators.get("donchian_high"),
                    "two_b_bottom": indicators.get("two_b_bottom"),
                    "two_b_top": indicators.get("two_b_top"),
                    "reason": original_guard.get("reason"),
                }
            )
        if vision_guard:
            strategy_events.append(
                {
                    "event_type": "vision_guard",
                    "action": vision_guard.get("action"),
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "box_low": indicators.get("donchian_low"),
                    "box_high": indicators.get("donchian_high"),
                    "two_b_bottom": indicators.get("two_b_bottom"),
                    "two_b_top": indicators.get("two_b_top"),
                    "reason": vision_guard.get("reason"),
                }
            )
        if extreme_guard:
            strategy_events.append(
                {
                    "event_type": "extreme_guard",
                    "action": extreme_guard.get("action"),
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "box_low": indicators.get("donchian_low"),
                    "box_high": indicators.get("donchian_high"),
                    "two_b_bottom": indicators.get("two_b_bottom"),
                    "two_b_top": indicators.get("two_b_top"),
                    "reason": extreme_guard.get("reason"),
                }
            )
        if exhaustion_guard:
            strategy_events.append(
                {
                    "event_type": "trend_exhaustion_guard",
                    "action": exhaustion_guard.get("action"),
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "box_low": indicators.get("donchian_low"),
                    "box_high": indicators.get("donchian_high"),
                    "two_b_bottom": indicators.get("two_b_bottom"),
                    "two_b_top": indicators.get("two_b_top"),
                    "reason": exhaustion_guard.get("reason"),
                }
            )
        if conflict_guard:
            strategy_events.append(
                {
                    "event_type": "trend_conflict_guard",
                    "action": conflict_guard.get("action"),
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "box_low": indicators.get("donchian_low"),
                    "box_high": indicators.get("donchian_high"),
                    "two_b_bottom": indicators.get("two_b_bottom"),
                    "two_b_top": indicators.get("two_b_top"),
                    "reason": conflict_guard.get("reason"),
                }
            )
        if rule_veto:
            strategy_events.append(
                {
                    "event_type": "rule_veto",
                    "action": rule_veto.get("action"),
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "reason": rule_veto.get("reason"),
                }
            )
        if reversal:
            strategy_events.append(
                {
                    "event_type": "reversal_signal",
                    "action": reversal.get("action"),
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "box_low": indicators.get("donchian_low"),
                    "box_high": indicators.get("donchian_high"),
                    "two_b_bottom": indicators.get("two_b_bottom"),
                    "two_b_top": indicators.get("two_b_top"),
                    "reason": reversal.get("reason"),
                }
            )
        quality = entry_quality(decision["action"], indicators)
        if quality:
            quality_action = decision["action"]
            decision = {
                **decision,
                "action": "WAIT",
                "confidence": min(
                    float(decision.get("confidence") or 0), 40
                ),
                "reason": str(quality["reason"])[:300],
                "add_on": False,
            }
            strategy_events.append(
                {
                    "event_type": "quality_block",
                    "action": quality_action,
                    "close": indicators.get("close"),
                    "rsi": indicators.get("rsi"),
                    "atr": indicators.get("atr"),
                    "box_low": indicators.get("donchian_low"),
                    "box_high": indicators.get("donchian_high"),
                    "two_b_bottom": indicators.get("two_b_bottom"),
                    "two_b_top": indicators.get("two_b_top"),
                    "reason": quality.get("reason"),
                }
            )
        decision["boundary_guard"] = guard
        decision["vision_guard"] = vision_guard
        decision["extreme_guard"] = extreme_guard
        decision["trend_exhaustion_guard"] = exhaustion_guard
        decision["trend_conflict_guard"] = conflict_guard
        decision["rule_veto"] = rule_veto
        decision["quality_gate"] = quality
        decision["strategy_events"] = strategy_events
        return decision
