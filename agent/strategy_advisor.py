"""Regime-aware strategy advisor.

Only suggests a preferred trading logic. The trader LLM keeps full autonomy
and may override, wait or take the opposite side.
"""

from __future__ import annotations


class StrategyAdvisor:
    def advise(self, market: dict, db=None) -> dict:
        indicators = market["indicators"]
        state = indicators.get("market_state", "range")
        confidence = indicators.get("state_confidence", "low")
        confidence_text = {"high": "较高", "medium": "中等", "low": "较低"}.get(
            confidence, "较低"
        )
        if state == "trend":
            default = {
                "market_state": "趋势",
                "state_confidence": confidence_text,
                "preferred_persona": "momentum",
                "alternatives": ["trend_follow", "breakout"],
                "logic": (
                    "顺势和动量延续优先，避免逆势抄底摸顶；"
                    "强趋势下允许顺势追入（突破新高/动量延续），"
                    "不必机械等待回调，回调可能不来；"
                    "入场必须带紧止损并控制仓位。"
                    f"当前状态判断置信度{confidence_text}，"
                    f"本轮momentum_ok={'是' if indicators.get('momentum_ok') else '否'}，"
                    "若为是，请直接给出顺势方向，不要再等待回调或突破确认；"
                    "若信号与建议冲突，以你自己的综合判断为准。"
                ),
            }
            candidates = ["momentum", "trend_follow", "breakout"]
        else:
            default = {
            "market_state": "震荡",
            "state_confidence": confidence_text,
            "preferred_persona": "mean_reversion",
            "alternatives": ["conservative"],
            "logic": (
                "箱体区间高抛低吸优先，避免追涨杀跌；"
                "只有突破被确认后才考虑转向趋势逻辑。"
                f"当前状态判断置信度{confidence_text}，"
                f"本轮range_signal={indicators.get('range_signal', 'none')}，"
                "若为BUY/SELL，请直接按均值回归方向执行，不要再等待；"
                    "若信号与建议冲突，以你自己的综合判断为准。"
                ),
            }
            candidates = ["mean_reversion", "conservative"]

        learned = self._learned_persona(db, state, candidates, default["preferred_persona"])
        result = dict(default)
        if learned != default["preferred_persona"]:
            result["preferred_persona"] = learned
            result["logic"] = (
                default["logic"]
                + f"；本轮根据最近复盘样本自动切换为 {learned} 优先。"
            )
        return result

    @staticmethod
    def _learned_persona(db, state, candidates, fallback) -> str:
        if db is None:
            return fallback
        rows = db.conn.execute(
            "SELECT market_state, persona, pnl FROM trades "
            "WHERE status='closed' AND market_state=? AND persona IS NOT NULL "
            "ORDER BY id DESC LIMIT 60",
            (state,),
        ).fetchall()
        stats: dict[str, dict] = {}
        for row in rows:
            entry = stats.setdefault(row["persona"], {"n": 0, "wins": 0, "net": 0.0})
            entry["n"] += 1
            pnl = float(row["pnl"] or 0.0)
            entry["net"] += pnl
            if pnl > 0:
                entry["wins"] += 1
        best = fallback
        best_score = float("-inf")
        for persona in candidates:
            entry = stats.get(persona)
            if not entry or entry["n"] < 5:
                continue
            score = (entry["wins"] - (entry["n"] - entry["wins"])) / entry["n"]
            score += min(entry["net"], 0.0) / 1000
            if score > best_score:
                best_score = score
                best = persona
        return best
