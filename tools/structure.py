"""Deterministic box-structure signal used to capture confirmed range entries."""

from __future__ import annotations

from config import (
    BOX_BOUNDARY_ATR,
    BOUNDARY_OVERBOUGHT_RSI,
    BOUNDARY_OVERSOLD_RSI,
    CLOSE_MIN_CONFIDENCE,
    CLOSE_RSI_OVERBOUGHT,
    CLOSE_RSI_OVERSOLD,
    ENABLE_AI_CLOSE,
    ENABLE_BOUNDARY_GUARD,
    ENABLE_EXTREME_GUARD,
    ENABLE_TREND_CONFLICT_GUARD,
    ENABLE_TREND_EXHAUSTION_GUARD,
    ENABLE_VISION_GUARD,
    EXTREME_EDGE_ATR,
    EXTREME_MOVE_ATR,
    EXTREME_RSI_BUY,
    EXTREME_RSI_SELL,
    REVERSAL_CONFIRM_ATR,
    REVERSAL_CONFIDENCE,
    REVERSAL_MAX_MOVE_ATR,
    STRUCTURE_CONFIDENCE,
    TREND_EXHAUSTION_ADX_MIN,
    TREND_EXHAUSTION_EDGE_ATR,
    TREND_EXHAUSTION_RSI_BUY_MIN,
    TREND_EXHAUSTION_RSI_SELL_MAX,
)


def _vision_confirms(vision: dict | None, direction: str) -> bool:
    if not vision:
        return False
    text = " ".join(
        str(vision.get(key, ""))
        for key in ("structure", "setup", "notes", "trend")
    ).lower()
    keywords = (
        ("2b", "假突破", "收回", "反转", "双底", "看涨", "bullish")
        if direction == "bottom"
        else ("2b", "假突破", "收回", "反转", "双顶", "看跌", "bearish")
    )
    return any(keyword in text for keyword in keywords)


def vision_reversal_guard(
    proposed_action: str,
    vision: dict | None,
) -> dict | None:
    """Hard block when the candle read says do not chase this direction.

    This turns prompt rule 14 into code: an explicit stop candle blocks new
    SELL entries, and an explicit stall candle blocks new BUY entries. The
    trader layer still decides whether an existing opposite-side position is
    closed first; this guard only stops adding to the blocked direction.
    """
    if not ENABLE_VISION_GUARD or not vision:
        return None
    action = str(proposed_action or "").upper()
    if action not in ("BUY", "SELL"):
        return None
    text = " ".join(
        str(vision.get(key, ""))
        for key in ("structure", "setup", "notes", "trend")
    )
    lower = text.lower()
    if action == "SELL" and any(
        keyword in text
        for keyword in (
            "止跌",
            "停止追空",
            "禁止追空",
            "禁止继续追空",
        )
    ):
        return {
            "action": action,
            "block": True,
            "reason": f"视觉K线保护：出现止跌信号（{text[:120]}），禁止继续追空",
        }
    if action == "BUY" and any(
        keyword in text
        for keyword in (
            "滞涨",
            "停止追多",
            "禁止追多",
            "禁止继续追多",
        )
    ):
        return {
            "action": action,
            "block": True,
            "reason": f"视觉K线保护：出现滞涨信号（{text[:120]}），禁止继续追多",
        }
    if action == "SELL" and (
        "bullish reversal" in lower
        or "bullish engulfing" in lower
        or "hammer" in lower
        or "stop candle" in lower
    ):
        return {
            "action": action,
            "block": True,
            "reason": f"视觉K线保护：模型识别到止跌K线（{text[:120]}），禁止继续追空",
        }
    if action == "BUY" and (
        "bearish reversal" in lower
        or "bearish engulfing" in lower
        or "shooting star" in lower
        or "stall candle" in lower
    ):
        return {
            "action": action,
            "block": True,
            "reason": f"视觉K线保护：模型识别到滞涨K线（{text[:120]}），禁止继续追多",
        }
    return None


def boundary_guard(
    proposed_action: str,
    indicators: dict,
    vision: dict | None = None,
) -> dict | None:
    """Hard block for entries that fight the box boundary.

    Priority is boundary protection > trend-following: a SELL inside the box
    bottom zone while RSI is oversold, or a BUY inside the box top zone while
    RSI is overbought, is blocked even when market_state says trend.
    """
    if not ENABLE_BOUNDARY_GUARD:
        return None
    action = str(proposed_action or "").upper()
    if action not in ("BUY", "SELL"):
        return None
    box_high = float(indicators.get("donchian_high") or 0)
    box_low = float(indicators.get("donchian_low") or 0)
    close = float(indicators.get("close") or 0)
    atr = float(indicators.get("atr") or 0)
    if atr <= 0 or box_high <= box_low or close <= 0:
        return None

    rsi = float(indicators.get("rsi") or 50)
    threshold = float(BOX_BOUNDARY_ATR)
    near_bottom = close <= box_low + threshold * atr
    near_top = close >= box_high - threshold * atr

    if action == "SELL" and near_bottom and rsi <= float(BOUNDARY_OVERSOLD_RSI):
        return {
            "action": action,
            "block": True,
            "reason": (
                f"箱体边界保护：价格 {close:.2f} 贴箱底 {box_low:.2f} "
                f"（{threshold:.1f}×ATR 内）且 RSI {rsi:.1f}≤{float(BOUNDARY_OVERSOLD_RSI):.0f}，"
                "禁止追空；只有 2B/假突破收回确认后才允许反向做多。"
            ),
        }
    if action == "BUY" and near_top and rsi >= float(BOUNDARY_OVERBOUGHT_RSI):
        return {
            "action": action,
            "block": True,
            "reason": (
                f"箱体边界保护：价格 {close:.2f} 贴箱顶 {box_high:.2f} "
                f"（{threshold:.1f}×ATR 内）且 RSI {rsi:.1f}≥{float(BOUNDARY_OVERBOUGHT_RSI):.0f}，"
                "禁止追多；只有 2B/假突破收回确认后才允许反向做空。"
            ),
        }
    return None


def extreme_chase_guard(
    proposed_action: str,
    indicators: dict,
) -> dict | None:
    """Block same-direction entries after an extended move into an extreme.

    Uses three conditions together: strong trend, RSI at the extreme, and price
    close to the recent boundary after a multi-ATR move. No single RSI value
    can block by itself.
    """
    if not ENABLE_EXTREME_GUARD:
        return None
    action = str(proposed_action or "").upper()
    if action not in ("BUY", "SELL"):
        return None
    close = float(indicators.get("close") or 0)
    atr = float(indicators.get("atr") or 0)
    rsi = float(indicators.get("rsi") or 50)
    market_state = str(indicators.get("market_state") or "range")
    trend = str(indicators.get("trend") or "neutral")
    donchian_high = float(indicators.get("donchian_high") or 0)
    donchian_low = float(indicators.get("donchian_low") or 0)
    closes = [float(v) for v in (indicators.get("closes") or []) if v]
    if atr <= 0 or donchian_low <= 0 or donchian_high <= donchian_low:
        return None
    recent = closes[-8:] or [close]
    recent_high = max(recent)
    recent_low = min(recent)
    edge = float(EXTREME_EDGE_ATR)
    move = float(EXTREME_MOVE_ATR)
    if (
        action == "SELL"
        and market_state == "trend"
        and trend == "bearish"
        and rsi <= float(EXTREME_RSI_SELL)
        and close <= donchian_low + edge * atr
        and recent_high - close >= move * atr
    ):
        return {
            "action": action,
            "block": True,
            "reason": (
                f"极端追单保护：空头趋势RSI {rsi:.1f}≤{float(EXTREME_RSI_SELL):.0f}，"
                f"价格 {close:.2f} 已贴近近期低点 {donchian_low:.2f}，"
                f"且从近期高点回落≥{move:.1f}×ATR，禁止继续追空"
            ),
        }
    if (
        action == "BUY"
        and market_state == "trend"
        and trend == "bullish"
        and rsi >= float(EXTREME_RSI_BUY)
        and close >= donchian_high - edge * atr
        and close - recent_low >= move * atr
    ):
        return {
            "action": action,
            "block": True,
            "reason": (
                f"极端追单保护：多头趋势RSI {rsi:.1f}≥{float(EXTREME_RSI_BUY):.0f}，"
                f"价格 {close:.2f} 已贴近近期高点 {donchian_high:.2f}，"
                f"且从近期低点上涨≥{move:.1f}×ATR，禁止继续追多"
            ),
        }
    return None


def trend_conflict_guard(
    proposed_action: str,
    indicators: dict,
) -> dict | None:
    """Block trend-state entries that fight the SMA trend without a reversal.

    When indicators.trend (SMA trend) is bullish/bearish but structure is the
    opposite, a new same-side-as-structure order is only allowed after a current
    CHoCH or a close through the recent Donchian extreme. A stall/stop candle
    alone is not a reversal confirmation.
    """
    if not ENABLE_TREND_CONFLICT_GUARD:
        return None
    action = str(proposed_action or "").upper()
    if action not in ("BUY", "SELL"):
        return None
    market_state = str(indicators.get("market_state") or "range")
    if market_state != "trend":
        return None
    major_trend = str(indicators.get("trend") or "neutral")
    structure = indicators.get("structure") or {}
    structure_trend = str(structure.get("trend") or "neutral")
    if major_trend not in ("bullish", "bearish"):
        return None
    structure_fights_action = (
        action == "BUY"
        and major_trend == "bullish"
        and structure_trend == "bearish"
    ) or (
        action == "SELL"
        and major_trend == "bearish"
        and structure_trend == "bullish"
    )
    if structure_fights_action:
        close = float(indicators.get("close") or 0)
        donchian_high = float(indicators.get("donchian_high") or 0)
        donchian_low = float(indicators.get("donchian_low") or 0)
        close_through_extreme = (
            action == "BUY" and close >= donchian_high
        ) or (action == "SELL" and close <= donchian_low)
        if not close_through_extreme:
            direction = "追多" if action == "BUY" else "追空"
            return {
                "action": action,
                "block": True,
                "reason": (
                    f"趋势冲突保护：均线趋势={major_trend}，"
                    f"但市场结构仍为{structure_trend}；禁止仅凭均线方向{direction}，"
                    "需等收盘突破/跌破Donchian极值、结构确认同向后再顺趋势开仓"
                ),
            }
    fights_major = (
        action == "SELL"
        and major_trend == "bullish"
        and structure_trend == "bearish"
    ) or (
        action == "BUY"
        and major_trend == "bearish"
        and structure_trend == "bullish"
    )
    if not fights_major:
        return None
    close = float(indicators.get("close") or 0)
    donchian_high = float(indicators.get("donchian_high") or 0)
    donchian_low = float(indicators.get("donchian_low") or 0)
    choch_up = bool(structure.get("choch_up"))
    choch_down = bool(structure.get("choch_down"))
    close_through_extreme = (
        action == "SELL"
        and donchian_low > 0
        and close <= donchian_low
    ) or (
        action == "BUY"
        and donchian_high > 0
        and close >= donchian_high
    )
    if not (choch_up or choch_down or close_through_extreme):
        return {
            "action": action,
            "block": True,
            "reason": (
                f"趋势冲突保护：均线趋势={major_trend}，结构趋势={structure_trend}，"
                f"当前无CHoCH/跌破极值确认，禁止在趋势状态逆{major_trend}开仓"
            ),
        }
    return None


def trend_exhaustion_guard(
    proposed_action: str,
    indicators: dict,
) -> dict | None:
    """Block new entries near a strong-trend extreme after an extended move."""
    if not ENABLE_TREND_EXHAUSTION_GUARD:
        return None
    action = str(proposed_action or "").upper()
    if action not in ("BUY", "SELL"):
        return None
    close = float(indicators.get("close") or 0)
    atr = float(indicators.get("atr") or 0)
    rsi = float(indicators.get("rsi") or 50)
    adx = float(indicators.get("adx") or 0)
    market_state = str(indicators.get("market_state") or "range")
    trend = str(indicators.get("trend") or "neutral")
    donchian_high = float(indicators.get("donchian_high") or 0)
    donchian_low = float(indicators.get("donchian_low") or 0)
    if atr <= 0 or donchian_low <= 0 or donchian_high <= donchian_low:
        return None
    edge = float(TREND_EXHAUSTION_EDGE_ATR)
    if (
        action == "SELL"
        and market_state == "trend"
        and trend == "bearish"
        and adx >= float(TREND_EXHAUSTION_ADX_MIN)
        and rsi <= float(TREND_EXHAUSTION_RSI_SELL_MAX)
        and close <= donchian_low + edge * atr
    ):
        return {
            "action": action,
            "block": True,
            "reason": (
                f"趋势末端保护：空头趋势ADX {adx:.1f}≥{float(TREND_EXHAUSTION_ADX_MIN):.0f}，"
                f"RSI {rsi:.1f}≤{float(TREND_EXHAUSTION_RSI_SELL_MAX):.0f}，"
                f"价格 {close:.2f} 距近期低点 {donchian_low:.2f} "
                f"不足 {edge:.1f}×ATR，即使动量确认也禁止追空"
            ),
        }
    if (
        action == "BUY"
        and market_state == "trend"
        and trend == "bullish"
        and adx >= float(TREND_EXHAUSTION_ADX_MIN)
        and rsi >= float(TREND_EXHAUSTION_RSI_BUY_MIN)
        and close >= donchian_high - edge * atr
    ):
        return {
            "action": action,
            "block": True,
            "reason": (
                f"趋势末端保护：多头趋势ADX {adx:.1f}≥{float(TREND_EXHAUSTION_ADX_MIN):.0f}，"
                f"RSI {rsi:.1f}≥{float(TREND_EXHAUSTION_RSI_BUY_MIN):.0f}，"
                f"价格 {close:.2f} 距近期高点 {donchian_high:.2f} "
                f"不足 {edge:.1f}×ATR，即使动量确认也禁止追多"
            ),
        }
    return None


def reversal_signal(
    indicators: dict,
    vision: dict | None = None,
) -> dict | None:
    """Confirmed reversal at a box extreme: 2B/false-breakout reclaim or vision."""
    box_high = float(indicators.get("donchian_high") or 0)
    box_low = float(indicators.get("donchian_low") or 0)
    close = float(indicators.get("close") or 0)
    atr = float(indicators.get("atr") or 0)
    if atr <= 0 or box_high <= box_low or close <= 0:
        return None

    threshold = float(REVERSAL_CONFIRM_ATR)
    rsi = float(indicators.get("rsi") or 50)
    if bool(indicators.get("two_b_bottom")) or _vision_confirms(vision, "bottom"):
        if close <= box_low + threshold * atr:
            return {
                "action": "BUY",
                "confidence": int(REVERSAL_CONFIDENCE),
                "reason": (
                    f"2B/假突破反转确认：价格 {close:.2f} 曾跌破箱底 {box_low:.2f} "
                    f"后收回，RSI {rsi:.1f} 处于低位，按反转做多，止损放箱底下方。"
                ),
            }
    if bool(indicators.get("two_b_top")) or _vision_confirms(vision, "top"):
        if close >= box_high - threshold * atr:
            return {
                "action": "SELL",
                "confidence": int(REVERSAL_CONFIDENCE),
                "reason": (
                    f"2B/假突破反转确认：价格 {close:.2f} 曾突破箱顶 {box_high:.2f} "
                    f"后收回，RSI {rsi:.1f} 处于高位，按反转做空，止损放箱顶上方。"
                ),
            }
    return None


def swing_reversal_signal(df, indicators: dict) -> dict | None:
    """Right-side swing reversal: pivot, reclaim, higher low, breakout.

    Fires only on the bar where the confirmation close just happened, so one
    pivot cannot generate repeated entries.
    """
    if df is None or len(df) < 12:
        return None
    atr = float(indicators.get("atr") or 0)
    if atr <= 0:
        return None
    n = min(14, len(df))
    lows = [float(v) for v in df["low"].tail(n)]
    highs = [float(v) for v in df["high"].tail(n)]
    closes = [float(v) for v in df["close"].tail(n)]
    if len(lows) < 12:
        return None
    max_move = float(REVERSAL_MAX_MOVE_ATR) * atr

    for p in range(n - 3, 1, -1):
        local_lows = lows[max(0, p - 2) : p + 3]
        if lows[p] > min(local_lows) or lows[p] > min(lows[: p + 1]):
            continue
        trigger = None
        for t in range(p + 1, n):
            if closes[t] > highs[p]:
                trigger = t
                break
        if trigger is None:
            continue
        confirmed = None
        for j in range(trigger + 1, n):
            if lows[j] > lows[p] and closes[j] > max(highs[trigger:j]):
                confirmed = j
                break
        if confirmed is None:
            continue
        if confirmed == n - 1 and closes[-1] - lows[p] <= max_move:
            return {
                "action": "BUY",
                "confidence": int(REVERSAL_CONFIDENCE),
                "reason": (
                    f"swing反转右侧确认：pivot低点 {lows[p]:.2f} 后出现更高低点，"
                    f"收盘 {closes[-1]:.2f} 突破局部高点，转为做多，"
                    "止损放pivot低点下方"
                ),
            }
        return None

    for p in range(n - 3, 1, -1):
        local_highs = highs[max(0, p - 2) : p + 3]
        if highs[p] < max(local_highs) or highs[p] < max(highs[: p + 1]):
            continue
        trigger = None
        for t in range(p + 1, n):
            if closes[t] < lows[p]:
                trigger = t
                break
        if trigger is None:
            continue
        confirmed = None
        for j in range(trigger + 1, n):
            if highs[j] < highs[p] and closes[j] < min(lows[trigger:j]):
                confirmed = j
                break
        if confirmed is None:
            continue
        if confirmed == n - 1 and highs[p] - closes[-1] <= max_move:
            return {
                "action": "SELL",
                "confidence": int(REVERSAL_CONFIDENCE),
                "reason": (
                    f"swing反转右侧确认：pivot高点 {highs[p]:.2f} 后出现更低高点，"
                    f"收盘 {closes[-1]:.2f} 跌破局部低点，转为做空，"
                    "止损放pivot高点上方"
                ),
            }
        return None
    return None


def structure_signal(
    indicators: dict,
    vision: dict | None = None,
) -> dict | None:
    """Return BUY/SELL only when price is at a box boundary with confirmation.

    Conditions are deliberately strict so the win rate and reward/risk profile
    stay intact: range state, close to the box edge, momentum confirmed and RSI
    not fighting the direction.
    """
    if indicators.get("market_state") != "range":
        return None
    box_high = float(indicators.get("donchian_high") or 0)
    box_low = float(indicators.get("donchian_low") or 0)
    close = float(indicators.get("close") or 0)
    atr = float(indicators.get("atr") or 0)
    if atr <= 0 or box_high <= box_low or close <= 0:
        return None
    if not indicators.get("momentum_ok"):
        return None

    rsi = float(indicators.get("rsi") or 50)
    threshold = float(BOX_BOUNDARY_ATR)
    near_top = close >= box_high - threshold * atr
    near_bottom = close <= box_low + threshold * atr

    if near_top and rsi >= 55:
        return {
            "action": "SELL",
            "confidence": int(STRUCTURE_CONFIDENCE),
            "reason": (
                f"K线箱体顶部确认：价格 {close:.2f} 距箱顶 {box_high:.2f} "
                f"在 {threshold:.2f}×ATR 内，动量确认且RSI {rsi:.1f} 不超卖，"
                "顺箱体做空，止损放箱顶上方。"
            ),
        }
    if near_bottom and rsi <= 55:
        return {
            "action": "BUY",
            "confidence": int(STRUCTURE_CONFIDENCE),
            "reason": (
                f"K线箱体底部确认：价格 {close:.2f} 距箱底 {box_low:.2f} "
                f"在 {threshold:.2f}×ATR 内，动量确认且RSI {rsi:.1f} 不过热，"
                "顺箱体做多，止损放箱底下方。"
            ),
        }
    return None


def close_signal(
    indicators: dict,
    positions: list | None = None,
    vision: dict | None = None,
) -> dict | None:
    """Multi-confirmation close signal for a held position.

    A close requires all of: the held side conflicts with the reversal,
    a confirmed 2B/false-breakout reclaim at the box boundary, price inside
    the boundary zone, and RSI leaning toward the reversal. No single
    indicator can trigger a close.
    """
    if not ENABLE_AI_CLOSE:
        return None
    position_list = positions or []
    if not position_list:
        return None
    box_high = float(indicators.get("donchian_high") or 0)
    box_low = float(indicators.get("donchian_low") or 0)
    close = float(indicators.get("close") or 0)
    atr = float(indicators.get("atr") or 0)
    rsi = float(indicators.get("rsi") or 50)
    sma20 = float(indicators.get("sma20") or 0)
    if atr <= 0 or box_high <= box_low or close <= 0:
        return None

    threshold = float(BOX_BOUNDARY_ATR)
    two_b_bottom = bool(indicators.get("two_b_bottom"))
    two_b_top = bool(indicators.get("two_b_top"))
    market_state = str(indicators.get("market_state") or "range")
    trend = str(indicators.get("trend") or "neutral")
    momentum_ok = bool(indicators.get("momentum_ok"))
    sell_positions = [
        pos for pos in position_list if str(pos.get("side", "")).upper() == "SELL"
    ]
    buy_positions = [
        pos for pos in position_list if str(pos.get("side", "")).upper() == "BUY"
    ]

    if (
        sell_positions
        and two_b_bottom
        and close <= box_low + threshold * atr
        and rsi <= float(CLOSE_RSI_OVERSOLD)
    ):
        return {
            "action": "CLOSE",
            "confidence": int(CLOSE_MIN_CONFIDENCE),
            "reason": (
                f"持仓反向确认：持有SELL但箱底出现2B假突破收回"
                f"（{close:.2f}≤箱底{box_low:.2f}+{threshold:.1f}×ATR），"
                f"RSI {rsi:.1f}≤{float(CLOSE_RSI_OVERSOLD):.0f}，趋势反转确认，"
                "无论浮亏浮盈均主动平掉SELL。"
            ),
        }
    if (
        buy_positions
        and two_b_top
        and close >= box_high - threshold * atr
        and rsi >= float(CLOSE_RSI_OVERBOUGHT)
    ):
        return {
            "action": "CLOSE",
            "confidence": int(CLOSE_MIN_CONFIDENCE),
            "reason": (
                f"持仓反向确认：持有BUY但箱顶出现2B假突破收回"
                f"（{close:.2f}≥箱顶{box_high:.2f}-{threshold:.1f}×ATR），"
                f"RSI {rsi:.1f}≥{float(CLOSE_RSI_OVERBOUGHT):.0f}，趋势反转确认，"
                "无论浮亏浮盈均主动平掉BUY。"
            ),
        }
    if (
        sell_positions
        and market_state == "trend"
        and trend == "bullish"
        and momentum_ok
        and rsi >= 55
        and sma20 > 0
        and close > sma20
    ):
        return {
            "action": "CLOSE",
            "confidence": int(CLOSE_MIN_CONFIDENCE),
            "reason": (
                f"持仓与趋势相反：持有SELL但市场为多头趋势（RSI {rsi:.1f}≥55，"
                f"收盘{close:.2f}>SMA20 {sma20:.2f}，动量确认），多重确认成立，"
                "主动平掉逆势SELL。"
            ),
        }
    if (
        buy_positions
        and market_state == "trend"
        and trend == "bearish"
        and momentum_ok
        and rsi <= 45
        and sma20 > 0
        and close < sma20
    ):
        return {
            "action": "CLOSE",
            "confidence": int(CLOSE_MIN_CONFIDENCE),
            "reason": (
                f"持仓与趋势相反：持有BUY但市场为空头趋势（RSI {rsi:.1f}≤45，"
                f"收盘{close:.2f}<SMA20 {sma20:.2f}，动量确认），多重确认成立，"
                "主动平掉逆势BUY。"
            ),
        }
    return None
