"""LLM clients.

RealDeepSeekLLM talks to any OpenAI-compatible API. MockLLM produces
deterministic offline answers so the full V1-V5 loop can be exercised
without credentials.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from config import (
    DEFAULT_RR,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_KEY,
    LLM_MODEL,
    LLM_TIMEOUT,
    USE_MOCK_LLM,
    VISION_API_KEY,
    VISION_BASE_URL,
    VISION_MODEL,
)


def extract_context(user_prompt: str) -> dict[str, Any]:
    """Extract the CONTEXT_JSON block embedded by agents."""
    match = re.search(r"CONTEXT_JSON:\s*(\{.*\})", user_prompt, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def parse_json(text: str) -> dict[str, Any]:
    """Parse the first balanced JSON object from a model answer."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"raw": text}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"raw": text}


class BaseLLM:
    def ask_text(self, system: str, user: str) -> str:
        raise NotImplementedError

    def ask_json(self, system: str, user: str) -> dict[str, Any]:
        raise NotImplementedError

    def ask_vision(self, image_path: str, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class DeepSeekLLM(BaseLLM):
    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=DEEPSEEK_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=LLM_TIMEOUT,
        )
        self.model = LLM_MODEL
        self.vision_model = VISION_MODEL

    def _chat(self, system: str, user: str, model: str | None = None) -> str:
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    def ask_text(self, system: str, user: str) -> str:
        return self._chat(system, user)

    def ask_json(self, system: str, user: str) -> dict[str, Any]:
        return parse_json(self._chat(system, user))

    def ask_vision(self, image_path: str, prompt: str) -> dict[str, Any]:
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        from openai import OpenAI

        client = OpenAI(
            api_key=VISION_API_KEY or DEEPSEEK_KEY,
            base_url=VISION_BASE_URL or DEEPSEEK_BASE_URL,
            timeout=LLM_TIMEOUT,
        )
        response = client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            temperature=0.2,
        )
        return parse_json(response.choices[0].message.content or "")


class MockLLM(BaseLLM):
    """Deterministic offline brain used by the demo."""

    def ask_text(self, system: str, user: str) -> str:
        if "review" in system.lower():
            return (
                "复盘摘要：纪律执行到位，止损被严格执行。"
                "经验教训：形态确认前不要加仓。"
            )
        return "模拟回复。"

    def ask_json(self, system: str, user: str) -> dict[str, Any]:
        ctx = extract_context(user)
        low_user = user.lower()
        if "trading decision" in low_user or "交易决策" in user:
            return self._trade(ctx)
        if (
            "视觉" in user
            or "chart" in low_user
            or "vision" in system.lower()
            or "image" in system.lower()
        ):
            return self._vision(ctx)
        if "event risk" in low_user or "宏观" in user:
            return self._macro(ctx)
        if "lesson" in low_user or "复盘" in user:
            return self._review(ctx)
        if "summarize trend" in low_user or "量化" in system:
            return self._market(ctx)
        return self._trade(ctx)

    def ask_vision(self, image_path: str, prompt: str) -> dict[str, Any]:
        return self._vision(extract_context(prompt))

    @staticmethod
    def _market(ctx: dict[str, Any]) -> dict[str, Any]:
        indicators = ctx.get("indicators", {})
        return {
            "trend": indicators.get("trend", ctx.get("trend", "neutral")),
            "momentum": "走强" if float(indicators.get("rsi", 50)) > 55 else "走弱",
            "volatility": indicators.get("volatility", "中等"),
            "summary": "上涨趋势，波动中等" if indicators.get("trend") == "bullish" else "下跌趋势，等待确认",
        }

    @staticmethod
    def _trade(ctx: dict[str, Any]) -> dict[str, Any]:
        trend = ctx.get("trend", "neutral")
        rsi = float(ctx.get("rsi", 50))
        close = float(ctx.get("close", 100))
        atr = max(float(ctx.get("atr", 1)), 0.01)
        advisor = ctx.get("strategy_advisor") or {}
        market_state = ctx.get("market_state") or advisor.get("market_state")
        range_signal = ctx.get("range_signal")
        momentum_ok = ctx.get("momentum_ok")
        positions = ctx.get("positions") or []

        action = "WAIT"
        confidence = 45
        add_on = False
        reason = f"趋势={trend}，RSI={rsi:.1f}，无明显优势"
        if market_state in ("震荡", "range") and range_signal in ("BUY", "SELL"):
            action = range_signal
            confidence = 70
            reason = (
                "震荡市边界信号触发，均值回归高抛"
                if action == "SELL"
                else "震荡市边界信号触发，均值回归低吸"
            )
        elif market_state == "趋势" and momentum_ok:
            action = "BUY" if trend == "bullish" else "SELL"
            confidence = 72
            reason = "趋势动量确认，顺势入场"
        elif market_state in ("震荡", "range"):
            if rsi >= 70:
                action, confidence = "SELL", 68
                reason = "震荡市RSI超买，均值回归高抛"
            elif rsi <= 30:
                action, confidence = "BUY", 68
                reason = "震荡市RSI超卖，均值回归低吸"
            else:
                action, confidence = "WAIT", 45
                reason = f"震荡市RSI={rsi:.1f}中性，等待边界信号"
        elif trend == "bullish" and rsi < 58:
            action, confidence = "BUY", 78
            reason = "上涨趋势且未超买，仍有空间"
        elif trend == "bearish" and rsi > 42:
            action, confidence = "SELL", 76
            reason = "下跌趋势且未超卖，仍有空间"
        elif trend == "bullish":
            action, confidence = "BUY", 66
            reason = "上涨趋势但RSI偏高"
        elif trend == "bearish":
            action, confidence = "SELL", 64
            reason = "下跌趋势但RSI偏低"

        sl_dist = max(atr * 1.5, close * 0.001)
        if action == "BUY":
            sl = round(close - sl_dist, 2)
            tp = round(close + sl_dist * float(DEFAULT_RR), 2)
        elif action == "SELL":
            sl = round(close + sl_dist, 2)
            tp = round(close - sl_dist * float(DEFAULT_RR), 2)
        else:
            sl = None
            tp = None
        close_hint = ctx.get("close_signal")
        if close_hint and positions:
            return {
                "action": "CLOSE",
                "confidence": int(close_hint.get("confidence") or 78),
                "reason": (
                    str(close_hint.get("reason") or "持仓反向多重确认，主动平仓")
                ),
                "sl": None,
                "tp": None,
                "risk_percent": 0.01,
                "add_on": False,
            }
        if action != "WAIT" and positions:
            same_side = any(
                str(pos.get("side", "")).upper() == action for pos in positions
            )
            if (
                market_state == "趋势"
                and momentum_ok
                and same_side
                and len(positions) < 3
            ):
                add_on = True
                confidence = max(confidence, 80)
                reason = "强趋势动量延续，AI判断允许顺势加仓"
            else:
                action, confidence = "WAIT", 35
                reason = "已有持仓且未达到加仓条件，等待趋势强化或平仓后再考虑"
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "sl": sl,
            "tp": tp,
            "risk_percent": 0.01,
            "add_on": add_on,
        }

    @staticmethod
    def _vision(ctx: dict[str, Any]) -> dict[str, Any]:
        closes = ctx.get("closes", [])
        trend = ctx.get("trend", "neutral")
        structure = "震荡"
        if len(closes) >= 10:
            highs = [v for v in closes[-10:] if v > closes[-10]]
            lows = [v for v in closes[-10:] if v < closes[-10]]
            if trend == "bullish":
                structure = "高点更高 低点更高" if highs else "上涨趋势中的回调"
            elif trend == "bearish":
                structure = "低点更低 高点更低" if lows else "下跌趋势中的反弹"
        confidence = 68 if trend != "neutral" else 52
        return {
            "trend": trend,
            "structure": structure,
            "setup": (
                "潜在做多"
                if trend == "bullish"
                else "潜在做空" if trend == "bearish" else "无形态"
            ),
            "confidence": confidence,
            "notes": "本地视觉分析",
        }

    @staticmethod
    def _macro(ctx: dict[str, Any]) -> dict[str, Any]:
        risk = ctx.get("event_risk", "low")
        return {
            "event_risk": risk,
            "recommendation": "正常" if risk == "low" else "减仓",
        }

    @staticmethod
    def _review(ctx: dict[str, Any]) -> dict[str, Any]:
        total = ctx.get("total", 0)
        wins = ctx.get("wins", 0)
        pnl = ctx.get("pnl", 0.0)
        return {
            "summary": f"共{total}笔交易，{wins}笔盈利，净盈亏{pnl:.2f}",
            "lesson": (
                "胜率和风险纪律比预测更重要。"
                "及时止损、绝不扩大止损，只交易有明确依据的形态。"
            ),
        }


def get_llm() -> BaseLLM:
    if USE_MOCK_LLM:
        return MockLLM()
    if not DEEPSEEK_KEY:
        raise RuntimeError("USE_MOCK_LLM=false requires DEEPSEEK_KEY")
    return DeepSeekLLM()
