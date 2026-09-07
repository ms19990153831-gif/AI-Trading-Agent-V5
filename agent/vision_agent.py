"""Vision agent: generates the candlestick chart and reads it with the LLM."""

from __future__ import annotations

import json

from brain.llm import get_llm
from vision.analyzer import VisionAnalyzer
from vision.chart import ChartGenerator


class VisionAgent:
    def __init__(self, llm=None, generator=None, analyzer=None) -> None:
        self.generator = generator or ChartGenerator()
        self.analyzer = analyzer or VisionAnalyzer(llm or get_llm())

    def observe(self, market: dict, bars: int = 100) -> dict:
        df = market["df"].tail(bars)
        chart_path = self.generator.create(
            df,
            symbol=market["symbol"],
            timeframe=market["timeframe"],
        )
        recent = df.tail(12)
        context = {
            "symbol": market["symbol"],
            "timeframe": market["timeframe"],
            "trend": market["indicators"]["trend"],
            "closes": market["indicators"]["closes"],
            "close": market["indicators"]["close"],
            "bars": [
                {
                    "time": str(row["time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
                for _, row in recent.iterrows()
            ],
        }
        vision = self.analyzer.analyze(
            chart_path,
            json.loads(json.dumps(context, ensure_ascii=False)),
        )
        return {"chart": chart_path, "vision": vision}
