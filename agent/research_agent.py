"""Research AI: scans the watchlist and ranks trading opportunities."""

from __future__ import annotations

import json

from brain.llm import get_llm
from brain.prompt import MARKET_SYSTEM
from config import BAR_COUNT, LITE_AGENTS, SYMBOLS, TIMEFRAME
from mt5.market import MarketData, compute_indicators

DEFAULT_WATCHLIST = SYMBOLS or ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]


class ResearchAgent:
    def __init__(
        self,
        llm=None,
        market_data=None,
        watchlist=None,
        timeframe: str | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.market_data = market_data or MarketData()
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.timeframe = timeframe or TIMEFRAME

    def scan(self, limit: int = 4, use_llm: bool | None = None) -> list[dict]:
        if use_llm is None:
            use_llm = not LITE_AGENTS
        findings: list[dict] = []
        for symbol in self.watchlist[:limit]:
            try:
                df = self.market_data.get_bars(
                    symbol, self.timeframe, BAR_COUNT
                )
                indicators = compute_indicators(df)
                if not use_llm:
                    view = self._local_view(indicators)
                    findings.append(
                        {
                            "symbol": symbol,
                            "score": view.get("opportunity_score", 50),
                            "trend": view.get("trend", indicators["trend"]),
                            "summary": view.get("summary", ""),
                            "indicators": {
                                key: value
                                for key, value in indicators.items()
                                if key != "closes"
                            },
                        }
                    )
                    continue
                context = {**indicators, "symbol": symbol}
                user = (
                    "CONTEXT_JSON: "
                    + json.dumps(context, ensure_ascii=False)
                    + "\nRate this symbol's trading opportunity (0-100) and "
                    "output strict JSON: trend, momentum, opportunity_score, summary."
                )
                try:
                    view = self.llm.ask_json(MARKET_SYSTEM, user)
                except Exception:
                    view = {
                        "trend": indicators["trend"],
                        "opportunity_score": 55,
                        "summary": "research llm unavailable",
                    }
                score = float(view.get("opportunity_score", 0) or 0)
                findings.append(
                    {
                        "symbol": symbol,
                        "score": round(score, 1),
                        "trend": view.get("trend", indicators["trend"]),
                        "summary": str(view.get("summary", ""))[:120],
                        "indicators": {
                            key: value
                            for key, value in indicators.items()
                            if key != "closes"
                        },
                    }
                )
            except Exception as exc:
                findings.append(
                    {
                        "symbol": symbol,
                        "score": 0,
                        "trend": "unknown",
                        "summary": f"scan error: {str(exc)[:80]}",
                    }
                )
        findings.sort(key=lambda item: item.get("score", 0), reverse=True)
        return findings

    @staticmethod
    def _local_view(indicators: dict) -> dict:
        trend = indicators.get("trend", "neutral")
        rsi = float(indicators.get("rsi", 50))
        adx = float(indicators.get("adx", 0))
        score = 50.0
        if trend == "bullish":
            score += 10.0
        elif trend == "bearish":
            score += 6.0
        if 55 <= rsi <= 78 or 22 <= rsi <= 45:
            score += 8.0
        if rsi >= 70 or rsi <= 30:
            score += 4.0
        if adx >= 25:
            score += 8.0
        score = round(min(score, 96), 1)
        trend_zh = {"bullish": "看涨", "bearish": "看跌", "neutral": "震荡"}
        return {
            "trend": trend,
            "opportunity_score": score,
            "summary": (
                f"本地扫描：趋势{trend_zh.get(trend, trend)}，"
                f"RSI {rsi:.1f}，ADX {adx:.1f}，机会评分 {score}"
            ),
        }
