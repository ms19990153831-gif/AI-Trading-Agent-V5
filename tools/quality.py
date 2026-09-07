"""Entry quality gate: keep M15 signals as selective as the H1 regime."""

from __future__ import annotations

from config import (
    ADX_TREND_MIN,
    BOX_BOUNDARY_ATR,
    ENABLE_QUALITY_GATE,
    ENABLE_HTF_FILTER,
    HTF_ADX_MIN,
    HTF_RSI_BUY_MIN,
    HTF_RSI_SELL_MAX,
    RANGE_BUY_RSI_MAX,
    RANGE_SELL_RSI_MIN,
    RSI_TREND_BUY_MAX,
    RSI_TREND_BUY_MIN,
    RSI_TREND_SELL_MAX,
    RSI_TREND_SELL_MIN,
)


def entry_quality(action: str, indicators: dict) -> dict | None:
    """Return a block reason when an entry does not meet the quality bar."""
    if not ENABLE_QUALITY_GATE:
        return None
    action = str(action or "").upper()
    if action not in ("BUY", "SELL"):
        return None

    close = float(indicators.get("close") or 0)
    rsi = float(indicators.get("rsi") or 50)
    atr = float(indicators.get("atr") or 0)
    adx = float(indicators.get("adx") or 0)
    box_high = float(indicators.get("donchian_high") or 0)
    box_low = float(indicators.get("donchian_low") or 0)
    if atr <= 0 or box_high <= box_low or close <= 0:
        return None

    threshold = float(BOX_BOUNDARY_ATR)
    near_top = close >= box_high - threshold * atr
    near_bottom = close <= box_low + threshold * atr
    market_state = str(indicators.get("market_state") or "range")
    trend = str(indicators.get("trend") or "neutral")
    two_b_bottom = bool(indicators.get("two_b_bottom"))
    two_b_top = bool(indicators.get("two_b_top"))

    if ENABLE_HTF_FILTER:
        htf_trend = str(indicators.get("htf_trend") or "neutral")
        htf_adx = float(indicators.get("htf_adx") or 0)
        htf_rsi = float(indicators.get("htf_rsi") or 50)
        if action == "BUY" and not (
            htf_trend == "bullish"
            and htf_adx >= float(HTF_ADX_MIN)
            and htf_rsi >= float(HTF_RSI_BUY_MIN)
        ):
            return {
                "block": True,
                "reason": (
                    f"大周期方向过滤：M15做多需H1看涨且ADX≥{float(HTF_ADX_MIN):.0f}"
                    f"、RSI≥{float(HTF_RSI_BUY_MIN):.0f}（当前{htf_trend}/"
                    f"ADX {htf_adx:.0f}/RSI {htf_rsi:.0f}）"
                ),
            }
        if action == "SELL" and not (
            htf_trend == "bearish"
            and htf_adx >= float(HTF_ADX_MIN)
            and htf_rsi <= float(HTF_RSI_SELL_MAX)
        ):
            return {
                "block": True,
                "reason": (
                    f"大周期方向过滤：M15做空需H1看跌且ADX≥{float(HTF_ADX_MIN):.0f}"
                    f"、RSI≤{float(HTF_RSI_SELL_MAX):.0f}（当前{htf_trend}/"
                    f"ADX {htf_adx:.0f}/RSI {htf_rsi:.0f}）"
                ),
            }

    # Confirmed 2B/false-breakout reversal at the boundary bypasses the
    # M15-level ADX/RSI/range checks, but still must pass the H1 direction.
    if (action == "BUY" and two_b_bottom and near_bottom) or (
        action == "SELL" and two_b_top and near_top
    ):
        return None

    if market_state == "trend":
        if action == "BUY":
            if trend != "bullish":
                return {
                    "block": True,
                    "reason": f"M15质量门槛：趋势状态为{trend}，禁止逆势做多",
                }
            if adx < float(ADX_TREND_MIN):
                return {
                    "block": True,
                    "reason": (
                        f"M15质量门槛：趋势做多要求ADX≥{float(ADX_TREND_MIN):.0f}，"
                        f"当前{adx:.1f}"
                    ),
                }
            if not (
                float(RSI_TREND_BUY_MIN)
                <= rsi
                <= float(RSI_TREND_BUY_MAX)
            ):
                return {
                    "block": True,
                    "reason": (
                        f"M15质量门槛：趋势做多RSI需在"
                        f"{float(RSI_TREND_BUY_MIN):.0f}-{float(RSI_TREND_BUY_MAX):.0f}，"
                        f"当前{rsi:.1f}"
                    ),
                }
        else:
            if trend != "bearish":
                return {
                    "block": True,
                    "reason": f"M15质量门槛：趋势状态为{trend}，禁止逆势做空",
                }
            if adx < float(ADX_TREND_MIN):
                return {
                    "block": True,
                    "reason": (
                        f"M15质量门槛：趋势做空要求ADX≥{float(ADX_TREND_MIN):.0f}，"
                        f"当前{adx:.1f}"
                    ),
                }
            if not (
                float(RSI_TREND_SELL_MIN)
                <= rsi
                <= float(RSI_TREND_SELL_MAX)
            ):
                return {
                    "block": True,
                    "reason": (
                        f"M15质量门槛：趋势做空RSI需在"
                        f"{float(RSI_TREND_SELL_MIN):.0f}-{float(RSI_TREND_SELL_MAX):.0f}，"
                        f"当前{rsi:.1f}"
                    ),
                }
    else:
        if action == "BUY" and not (
            near_bottom and rsi <= float(RANGE_BUY_RSI_MAX)
        ):
            return {
                "block": True,
                "reason": (
                    "M15质量门槛：震荡做多需贴箱底（0.6×ATR内）且"
                    f"RSI≤{float(RANGE_BUY_RSI_MAX):.0f}"
                ),
            }
        if action == "SELL" and not (
            near_top and rsi >= float(RANGE_SELL_RSI_MIN)
        ):
            return {
                "block": True,
                "reason": (
                    "M15质量门槛：震荡做空需贴箱顶（0.6×ATR内）且"
                    f"RSI≥{float(RANGE_SELL_RSI_MIN):.0f}"
                ),
            }
    return None
