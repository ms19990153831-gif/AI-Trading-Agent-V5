"""Market agent: observes the symbol and produces a numeric market view."""

from __future__ import annotations

import json

from brain.llm import get_llm
from brain.prompt import MARKET_SYSTEM
from config import ENABLE_HTF_FILTER, LITE_AGENTS, SYMBOL, TIMEFRAME
from mt5.market import (
    MarketData,
    build_context,
    compute_indicators,
    htf_snapshot_from_df,
)
from tools.market_structure import structure_reference_text


class MarketAgent:
    def __init__(
        self,
        market_data=None,
        llm=None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.market_data = market_data or MarketData()
        self.llm = llm or get_llm()
        self.symbol = symbol or SYMBOL
        self.timeframe = timeframe or TIMEFRAME

    def observe(self) -> dict:
        df = self.market_data.get_bars(self.symbol, self.timeframe)
        htf_snapshot = None
        if ENABLE_HTF_FILTER:
            try:
                htf_timeframe = (
                    "H1" if self.timeframe not in ("H1", "H4", "D1") else "H4"
                )
                htf_df = self.market_data.get_bars(
                    self.symbol, htf_timeframe, 200
                )
                htf_snapshot = htf_snapshot_from_df(htf_df)
            except Exception:
                htf_snapshot = None
        indicators = compute_indicators(df, htf=htf_snapshot)
        context = build_context(df, indicators)
        if LITE_AGENTS:
            trend_zh = {"bullish": "看涨", "bearish": "看跌", "neutral": "震荡"}
            structure_ref = structure_reference_text(
                indicators.get("structure")
            )
            momentum_ref = indicators.get("momentum_reference") or "暂无动量背离"
            key_ref = indicators.get("key_level_reference") or "暂无关键位"
            cci_ref = indicators.get("cci_reference") or "暂无CCI背离"
            ict_ref = indicators.get("ict_reference") or "暂无ICT结构"
            macd_ref = indicators.get("macd_reference") or "暂无MACD状态"
            phase_ref = indicators.get("trend_phase_reference") or "暂无阶段"
            view = {
                "trend": indicators["trend"],
                "momentum": "走强" if indicators["rsi"] > 55 else "走弱",
                "volatility": indicators.get("volatility", "中等"),
                "summary": (
                    f"趋势{trend_zh.get(indicators['trend'], indicators['trend'])}，"
                    f"RSI {indicators['rsi']}，ATR {indicators['atr']}；"
                    f"{structure_ref}；{momentum_ref}；{key_ref}；{cci_ref}；"
                    f"{ict_ref}；{macd_ref}；{phase_ref}"
                ),
            }
        else:
            user = (
                "CONTEXT_JSON: "
                + json.dumps(context, ensure_ascii=False)
                + "\nSummarize trend, momentum, volatility and structure."
            )
            try:
                view = self.llm.ask_json(MARKET_SYSTEM, user)
            except Exception as exc:
                view = {
                    "summary": f"market llm unavailable: {type(exc).__name__}",
                    "trend": indicators["trend"],
                }
        if "raw" in view:
            view = {"summary": str(view.get("raw"))[:300]}
        if not view.get("summary"):
            trend_zh = {"bullish": "看涨", "bearish": "看跌", "neutral": "震荡"}
            structure_ref = structure_reference_text(
                indicators.get("structure")
            )
            momentum_ref = indicators.get("momentum_reference") or "暂无动量背离"
            key_ref = indicators.get("key_level_reference") or "暂无关键位"
            cci_ref = indicators.get("cci_reference") or "暂无CCI背离"
            ict_ref = indicators.get("ict_reference") or "暂无ICT结构"
            macd_ref = indicators.get("macd_reference") or "暂无MACD状态"
            phase_ref = indicators.get("trend_phase_reference") or "暂无阶段"
            view["summary"] = (
                f"趋势{trend_zh.get(indicators['trend'], indicators['trend'])}，"
                f"RSI {indicators['rsi']}，"
                f"ATR {indicators['atr']}；{structure_ref}；{momentum_ref}；"
                f"{key_ref}；{cci_ref}；{ict_ref}；{macd_ref}；{phase_ref}"
            )
        return {
            "df": df,
            "indicators": indicators,
            "view": view,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "close": indicators["close"],
            "atr": indicators["atr"],
        }
