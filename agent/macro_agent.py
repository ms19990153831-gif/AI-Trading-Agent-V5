"""Macro agent: live gold/USD headlines plus event-risk assessment."""

from __future__ import annotations

import json
import time

from brain.llm import get_llm
from brain.prompt import MACRO_SYSTEM
from config import MACRO_CACHE_SECONDS, MACRO_RISK
from tools.news import fetch_market_news


class MacroAgent:
    def __init__(self, llm=None, event_risk: str | None = None) -> None:
        self.llm = llm or get_llm()
        self.event_risk = event_risk or MACRO_RISK
        self._cache: dict | None = None
        self._cached_at = 0.0

    def observe(self) -> dict:
        now = time.time()
        if self._cache and now - self._cached_at < MACRO_CACHE_SECONDS:
            return self._cache
        try:
            news = fetch_market_news()
        except Exception as exc:
            news = []
            self.event_risk = self.event_risk or "low"
        if not news:
            result = {
                "event_risk": self.event_risk,
                "recommendation": "normal",
                "summary": "暂无实时黄金/美联储相关新闻，按本地技术面执行",
                "headlines": [],
                "news_source": "none",
            }
            self._cache = result
            self._cached_at = now
            return result

        headlines = [
            {
                "source": item.get("source"),
                "time": item.get("time"),
                "text": item.get("text", "")[:220],
            }
            for item in news[:6]
        ]
        result = self._assess(headlines)
        result["headlines"] = headlines
        result["news_source"] = "live"
        self._cache = result
        self._cached_at = now
        return result

    def _assess(self, headlines: list[dict]) -> dict:
        user = (
            "CONTEXT_JSON: "
            + json.dumps({"headlines": headlines}, ensure_ascii=False)
            + "\nAssess event risk for XAUUSD trading. Be specific about "
            "which headline matters and what it means for gold."
        )
        try:
            result = self.llm.ask_json(MACRO_SYSTEM, user)
            if (
                "raw" not in result
                and result.get("event_risk") in ("low", "medium", "high")
                and result.get("recommendation")
                in ("normal", "reduce_position", "stand_down")
            ):
                if not result.get("summary"):
                    result["summary"] = self._summary(headlines)
                return result
        except Exception:
            pass
        return self._heuristic(headlines)

    def _heuristic(self, headlines: list[dict]) -> dict:
        text = " ".join(item.get("text", "") for item in headlines)
        high_markers = (
            "FOMC",
            "非农",
            "CPI",
            "PCE",
            "利率决议",
            "鲍威尔",
            "美联储",
            "Fed",
        )
        strong_markers = (
            "超预期",
            "意外",
            "鹰派",
            "鸽派",
            "紧急",
            "战争",
            "冲突",
            "地缘",
            "暴涨",
            "暴跌",
            "大幅",
        )
        risk_markers = (
            "降息",
            "加息",
            "通胀",
            "衰退",
            "美元",
            "美债",
            "收益率",
            "关税",
            "制裁",
            "避险",
        )
        high_hit = any(marker.lower() in text.lower() for marker in high_markers)
        strong_hit = any(marker.lower() in text.lower() for marker in strong_markers)
        risk_hit = any(marker.lower() in text.lower() for marker in risk_markers)
        if high_hit and strong_hit:
            event_risk = "high"
            recommendation = "reduce_position"
        elif high_hit or (risk_hit and strong_hit):
            event_risk = "medium"
            recommendation = "normal"
        else:
            event_risk = "low"
            recommendation = "normal"
        return {
            "event_risk": event_risk,
            "recommendation": recommendation,
            "summary": self._summary(headlines),
        }

    @staticmethod
    def _summary(headlines: list[dict]) -> str:
        if not headlines:
            return "暂无实时相关新闻"
        return "；".join(
            f"{item.get('text', '')[:100]}" for item in headlines[:3]
        )
