"""Vision analysis: send the chart image to the vision model, or use the
offline mock when no API key is configured."""

from __future__ import annotations

import json

from brain.llm import get_llm
from brain.prompt import VISION_SYSTEM
from config import VISION_API_KEY, VISION_MODEL


class VisionAnalyzer:
    def __init__(self, llm=None) -> None:
        self.llm = llm or get_llm()

    def analyze(self, image_path: str, context: dict) -> dict:
        prompt = (
            VISION_SYSTEM
            + "\n\nCONTEXT_JSON: "
            + json.dumps(context, ensure_ascii=False)
            + "\n\nAnalyze the attached chart and return strict JSON."
        )
        if not VISION_MODEL or not VISION_API_KEY:
            result = self._local_analysis(context)
            result["source"] = "local"
            result["notes"] = (
                "本地K线引擎：未配置可用的视觉模型，"
                "已用完整OHLC做K线结构分析"
            )
            return result
        try:
            result = self.llm.ask_vision(image_path, prompt)
        except Exception as exc:
            result = self._local_analysis(context)
            result["source"] = "local"
            result["notes"] = (
                f"{result.get('notes', '本地K线分析')}；视觉模型调用失败: "
                f"{type(exc).__name__}"
            )
            return result
        if "raw" in result:
            result = self._local_analysis(context)
            result["source"] = "local"
            return result
        result["source"] = "vision_model"
        return result

    @staticmethod
    def _local_analysis(context: dict) -> dict:
        """Candle-level chart read when no vision API is available."""
        closes = context.get("closes", [])
        trend = context.get("trend", "neutral")
        bars = context.get("bars") or []

        def _to_float(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        if len(bars) >= 4:
            opens = [_to_float(bar["open"]) for bar in bars]
            highs = [_to_float(bar["high"]) for bar in bars]
            lows = [_to_float(bar["low"]) for bar in bars]
            closes = [_to_float(bar["close"]) for bar in bars]

            def pattern(i: int) -> tuple[bool, bool]:
                o, h, l, c = opens[i], highs[i], lows[i], closes[i]
                candle_range = max(h - l, 1e-9)
                lower_wick = min(o, c) - l
                upper_wick = h - max(o, c)
                close_upper = c >= l + candle_range * 0.55
                close_lower = c <= h - candle_range * 0.45
                prior_lows = lows[max(0, i - 7) : i]
                prior_highs = highs[max(0, i - 7) : i]
                near_low = prior_lows and l <= min(prior_lows) * 1.002
                near_high = prior_highs and h >= max(prior_highs) * 0.998
                long_lower = lower_wick >= candle_range * 0.5
                long_upper = upper_wick >= candle_range * 0.5
                if i > 0:
                    prev_o = opens[i - 1]
                    prev_c = closes[i - 1]
                    prev_bearish = prev_c < prev_o
                    prev_bullish = prev_c > prev_o
                    bullish_engulf = (
                        prev_bearish and o <= prev_c and c > prev_o
                    )
                    bearish_engulf = (
                        prev_bullish and o >= prev_c and c < prev_o
                    )
                else:
                    bullish_engulf = False
                    bearish_engulf = False
                stop = near_low and (
                    long_lower or bullish_engulf
                ) and close_upper
                stall = near_high and (
                    long_upper or bearish_engulf
                ) and close_lower
                return bool(stop), bool(stall)

            active_stop_age = None
            active_stall_age = None
            for i in range(len(bars) - 1, 0, -1):
                stop, stall = pattern(i)
                if not (stop or stall):
                    continue
                later_closes = closes[i + 1 :] or []
                if stop and later_closes:
                    if min(later_closes) < lows[i] * 0.9995:
                        stop = False
                if stall and later_closes:
                    if max(later_closes) > highs[i] * 1.0005:
                        stall = False
                if not (stop or stall):
                    continue
                age = len(bars) - 1 - i
                if stop:
                    active_stop_age = (
                        age if active_stop_age is None else min(active_stop_age, age)
                    )
                if stall:
                    active_stall_age = (
                        age if active_stall_age is None else min(active_stall_age, age)
                    )

            if trend == "bearish":
                if active_stop_age is not None:
                    return {
                        "trend": "bearish",
                        "structure": "H1止跌K线（长下影/阳包阴）",
                        "setup": "停止追空，观察企稳",
                        "confidence": 74 if active_stop_age <= 2 else 70,
                        "notes": (
                            "本地K线分析：近期低点止跌信号仍有效，"
                            "后续反弹滞涨不改变禁止追空的判断"
                        ),
                    }
                if active_stall_age is not None:
                    return {
                        "trend": "bearish",
                        "structure": "反弹滞涨（长上影/阴包阴）",
                        "setup": "反弹结束，可顺势偏空",
                        "confidence": 70,
                        "notes": (
                            "本地K线分析：反弹在压力位滞涨，空头可重新关注"
                        ),
                    }
            else:
                if active_stall_age is not None:
                    return {
                        "trend": "bullish",
                        "structure": "H1滞涨K线（长上影/阴包阴）",
                        "setup": "停止追多，观察回落",
                        "confidence": 74 if active_stall_age <= 2 else 70,
                        "notes": (
                            "本地K线分析：近期高点滞涨信号仍有效，"
                            "后续回踩止跌不改变禁止追多的判断"
                        ),
                    }
                if active_stop_age is not None:
                    return {
                        "trend": "bullish",
                        "structure": "上涨回调止跌（长下影/阳包阴）",
                        "setup": "潜在做多",
                        "confidence": 72,
                        "notes": "本地K线分析：回调中出现止跌K线，方向偏多",
                    }

        if len(bars) >= 3:
            lows = [_to_float(bar["low"]) for bar in bars]
            closes = [_to_float(bar["close"]) for bar in bars]
            opens = [_to_float(bar["open"]) for bar in bars]
            highs = [_to_float(bar["high"]) for bar in bars]
            last = bars[-1]
            prev = bars[-2]
            o = _to_float(last["open"])
            h = _to_float(last["high"])
            l = _to_float(last["low"])
            c = _to_float(last["close"])
            candle_range = max(h - l, 1e-9)
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)
            prior_lows = lows[:-1]
            prior_highs = highs[:-1]
            recent_low_ref = (
                min(prior_lows[-6:]) if len(prior_lows) >= 1 else l
            )
            recent_high_ref = (
                max(prior_highs[-6:]) if len(prior_highs) >= 1 else h
            )
            near_recent_low = l <= recent_low_ref * 1.002
            near_recent_high = h >= recent_high_ref * 0.998
            long_lower_shadow = lower_wick >= candle_range * 0.5
            long_upper_shadow = upper_wick >= candle_range * 0.5
            close_in_upper_half = c >= l + candle_range * 0.55
            close_in_lower_half = c <= h - candle_range * 0.45

            prev_bearish = _to_float(prev["close"]) < _to_float(prev["open"])
            prev_bullish = _to_float(prev["close"]) > _to_float(prev["open"])
            bullish_engulf = (
                prev_bearish
                and o <= _to_float(prev["close"])
                and c > _to_float(prev["open"])
                and c > _to_float(prev["close"])
            )
            bearish_engulf = (
                prev_bullish
                and o >= _to_float(prev["close"])
                and c < _to_float(prev["open"])
                and c < _to_float(prev["close"])
            )
            stop_signal = (
                near_recent_low
                and (long_lower_shadow or bullish_engulf)
                and close_in_upper_half
            )
            stall_signal = (
                near_recent_high
                and (long_upper_shadow or bearish_engulf)
                and close_in_lower_half
            )
            if stop_signal and trend == "bearish":
                return {
                    "trend": "bearish",
                    "structure": "H1止跌K线（长下影/阳包阴）",
                    "setup": "停止追空，观察企稳",
                    "confidence": 74,
                    "notes": (
                        "本地K线分析：价格在近期低点出现长下影或阳包阴，"
                        "收盘回上半区，是明确止跌信号，禁止继续追空"
                    ),
                }
            if stop_signal and trend == "bullish":
                return {
                    "trend": "bullish",
                    "structure": "上涨回调止跌（长下影/阳包阴）",
                    "setup": "潜在做多",
                    "confidence": 72,
                    "notes": "本地K线分析：回调中出现止跌K线，方向偏多",
                }
            if stall_signal and trend == "bullish":
                return {
                    "trend": "bullish",
                    "structure": "H1滞涨K线（长上影/阴包阴）",
                    "setup": "停止追多，观察回落",
                    "confidence": 74,
                    "notes": (
                        "本地K线分析：价格在近期高点出现长上影或阴包阴，"
                        "收盘回下半区，是明确滞涨信号，禁止继续追多"
                    ),
                }
            if stall_signal and trend == "bearish":
                return {
                    "trend": "bearish",
                    "structure": "反弹滞涨（长上影/阴包阴）",
                    "setup": "反弹结束，可顺势偏空",
                    "confidence": 70,
                    "notes": "本地K线分析：反弹在压力位滞涨，空头可重新关注",
                }
            if trend == "bearish" and lower_wick >= candle_range * 0.3:
                return {
                    "trend": "bearish",
                    "structure": "下跌中出现下影试探",
                    "setup": "无形态",
                    "confidence": 58,
                    "notes": "本地K线分析：下跌中有下影线，追空需谨慎",
                }
            if trend == "bullish" and upper_wick >= candle_range * 0.3:
                return {
                    "trend": "bullish",
                    "structure": "上涨中出现上影试探",
                    "setup": "无形态",
                    "confidence": 58,
                    "notes": "本地K线分析：上涨中有上影线，追多需谨慎",
                }

        closes = context.get("closes") or closes
        structure = "震荡"
        setup = "无形态"
        confidence = 52
        if len(closes) >= 16:
            window = closes[-16:]
            highs = [
                max(float(window[i]), float(window[i + 1]))
                for i in range(0, len(window) - 1, 3)
            ]
            lows = [
                min(float(window[i]), float(window[i + 1]))
                for i in range(0, len(window) - 1, 3)
            ]
            higher_high = len(highs) >= 2 and highs[-1] > highs[-2]
            higher_low = len(lows) >= 2 and lows[-1] > lows[-2]
            if trend == "bullish" and higher_high and higher_low:
                structure = "高点更高 低点更高"
                setup = "潜在做多"
                confidence = 74
            elif trend == "bearish" and not higher_high and not higher_low:
                structure = "低点更低 高点更低"
                setup = "潜在做空"
                confidence = 72
            elif trend == "bullish":
                structure = "上涨趋势中的回调"
                confidence = 62
            elif trend == "bearish":
                structure = "下跌趋势中的反弹"
                confidence = 60
        return {
            "trend": trend,
            "structure": structure,
            "setup": setup,
            "confidence": confidence,
            "notes": "本地K线分析",
        }
