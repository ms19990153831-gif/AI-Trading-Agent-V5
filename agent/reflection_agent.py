"""Reflection agent: daily review and lesson extraction (V4)."""

from __future__ import annotations

import json

from brain.llm import get_llm
from brain.prompt import REVIEW_SYSTEM


def summarize_strategy_events(events: list[dict]) -> str:
    """Compact Chinese summary of boundary-guard and 2B reversal events."""
    if not events:
        return "无"
    guards = [e for e in events if e.get("event_type") == "boundary_guard"]
    reversals = [e for e in events if e.get("event_type") == "reversal_signal"]
    quality_blocks = [
        e for e in events if e.get("event_type") == "quality_block"
    ]
    ai_closes = [e for e in events if e.get("event_type") == "ai_close"]
    vision_guards = [e for e in events if e.get("event_type") == "vision_guard"]
    extreme_guards = [e for e in events if e.get("event_type") == "extreme_guard"]
    rule_vetos = [e for e in events if e.get("event_type") == "rule_veto"]
    blocked_reversals = [
        e for e in reversals if e.get("outcome") == "blocked"
    ]
    executed_reversals = [
        e for e in reversals if e.get("outcome") == "executed"
    ]
    parts = []
    if guards:
        sell_guards = sum(
            1 for e in guards if str(e.get("action", "")).upper() == "SELL"
        )
        buy_guards = len(guards) - sell_guards
        parts.append(
            f"边界保护拦截{len(guards)}次"
            f"（禁追空{sell_guards}次、禁追多{buy_guards}次）"
        )
    if reversals:
        parts.append(
            f"2B反转信号{len(reversals)}次"
            f"（成交{len(executed_reversals)}次、被持仓或风控挡住"
            f"{len(blocked_reversals)}次）"
        )
    if quality_blocks:
        parts.append(f"质量门槛拦截{len(quality_blocks)}次")
    if vision_guards:
        parts.append(f"视觉止跌/滞涨参考{len(vision_guards)}次")
    if extreme_guards:
        parts.append(f"极端追单保护{len(extreme_guards)}次")
    if ai_closes:
        parts.append(f"AI主动平仓{len(ai_closes)}次")
    if rule_vetos:
        parts.append(f"规则复核否决{len(rule_vetos)}次")
    return "；".join(parts)


class ReflectionAgent:
    def __init__(self, llm=None) -> None:
        self.llm = llm or get_llm()

    def review(
        self,
        trades: list[dict],
        extra: str | None = None,
        technical: str | None = None,
        macro: str | None = None,
        strategy_events: list[dict] | None = None,
        decisions: list[dict] | None = None,
    ) -> dict:
        wins = sum(1 for trade in trades if float(trade.get("pnl") or 0) > 0)
        total = len(trades)
        pnl = sum(float(trade.get("pnl") or 0) for trade in trades)
        closed = [t for t in trades if t.get("status") == "closed"]
        realized = sum(float(t.get("pnl") or 0) for t in closed)
        events = strategy_events or []
        decisions = decisions or []
        wait_actions = [
            d for d in decisions if str(d.get("action", "")).upper() == "WAIT"
        ]
        context = {
            "total": total,
            "closed": len(closed),
            "wins": wins,
            "realized_pnl": round(realized, 2),
            "open_pnl_est": round(pnl - realized, 2),
            "trades": trades[-10:],
            "technical_context": technical,
            "macro_context": macro,
            "strategy_events": events[-20:],
            "strategy_events_summary": summarize_strategy_events(events),
            "decisions": decisions[-40:],
            "decision_counts": {
                "total": len(decisions),
                "wait": len(wait_actions),
                "buy_or_sell": sum(
                    1
                    for d in decisions
                    if str(d.get("action", "")).upper() in ("BUY", "SELL")
                ),
                "close": sum(
                    1
                    for d in decisions
                    if str(d.get("action", "")).upper() == "CLOSE"
                ),
            },
        }
        user = (
            "CONTEXT_JSON: "
            + json.dumps(context, ensure_ascii=False)
            + ("\nMT5摘要: " + extra if extra else "")
            + "\n复盘必须结合K线、技术面、消息面和 strategy_events（视觉止跌/滞涨参考、"
            "边界保护拦截与2B反转信号），"
            "并检查 decisions 中的 WAIT 理由：是否因为堆叠过多参考而错过行情，"
            "或引用了失效/过期信号；找出当天空仓或反复等待的主要模式，"
            "写出一条能直接落到代码或规则上的可执行经验。"
        )
        try:
            result = self.llm.ask_json(REVIEW_SYSTEM, user)
        except Exception:
            result = {
                "summary": "llm unavailable during review",
                "lesson": "keep risk discipline and inspect trade logs",
            }
        if "raw" in result:
            return {
                "summary": "review produced no structured output",
                "lesson": "inspect trade logs before the next session",
            }
        return result
