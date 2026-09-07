"""Read-only market-structure context: swings, BOS, CHoCH and trend health.

The formulas are a Python port of the TradingView open-source indicator
"Market Structure - BOS, CHoCH, HH/HL/LH/LL & Trend Health [LunqFX]".
This module is context only: it never opens, blocks or closes a trade.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


_TREND_ZH = {"bullish": "多头", "bearish": "空头", "neutral": "中性"}
_HEALTH_STATE_ZH = {
    "STRONG": "强",
    "HOLDING": "尚可",
    "WEAKENING": "转弱",
    "NEW_TREND": "新趋势",
    "NO_STRUCTURE": "无明确结构",
}


def structure_reference_text(structure: dict[str, Any] | None) -> str:
    """Chinese one-line structure reference used by the AI context/review."""
    if not structure or structure.get("trend") == "neutral":
        return "暂无明确市场结构（BOS/CHoCH/健康度待确认）"
    trend = _TREND_ZH.get(str(structure.get("trend")), str(structure.get("trend")))
    health = structure.get("health")
    if health is not None:
        state = _HEALTH_STATE_ZH.get(
            str(structure.get("health_state")),
            str(structure.get("health_state")),
        )
        health_text = f"{health}/100（{state}）"
    else:
        health_text = "暂无健康度"
    tape = "；".join(str(item) for item in (structure.get("tape") or []))
    last_bar = []
    if structure.get("bos_up"):
        last_bar.append("本根收盘向上突破结构（BOS续势）")
    if structure.get("bos_down"):
        last_bar.append("本根收盘向下突破结构（BOS续势）")
    if structure.get("choch_up"):
        last_bar.append("本根收盘向上反转确认（CHoCH）")
    if structure.get("choch_down"):
        last_bar.append("本根收盘向下反转确认（CHoCH）")
    text = (
        f"市场结构趋势={trend}，健康度={health_text}，"
        f"最近摆动高={structure.get('swing_high')}，"
        f"最近摆动低={structure.get('swing_low')}"
    )
    if last_bar:
        text += f"；{'、'.join(last_bar)}"
    if tape:
        text += f"；近期结构事件={tape}"
    return text


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _atr_14(df: pd.DataFrame) -> float:
    """Simple 14-bar Wilder ATR used only when an atr is not supplied."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    value = float(true_range.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    return value if value == value else 0.0


def structure_snapshot(
    df: pd.DataFrame,
    swing_len: int = 10,
    atr: float | None = None,
    max_events: int = 5,
    weak_level: int = 40,
) -> dict[str, Any]:
    """Return a non-repainting structure read at the last completed bar.

    The port keeps Pine's event order: a newly confirmed pivot becomes the
    active level before the same bar checks for a break, and a break level is
    consumed until the next same-side pivot confirms.
    """
    if df is None or len(df) < 2 * swing_len + 1:
        return _empty_structure(swing_len)

    highs = [float(v) for v in df["high"]]
    lows = [float(v) for v in df["low"]]
    closes = [float(v) for v in df["close"]]
    times = [str(v) for v in df["time"]]
    n = len(highs)
    atr_value = float(atr) if atr is not None else _atr_14(df)
    if atr_value != atr_value or atr_value <= 0:
        atr_value = 0.0

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    high_labels: list[dict[str, Any]] = []
    low_labels: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    struct_high: float | None = None
    struct_low: float | None = None
    high_used = False
    low_used = False
    trend = 0  # -1 bearish, 0 neutral, 1 bullish
    leg_count = 0
    last_bos_up = False
    last_bos_down = False
    last_choch_up = False
    last_choch_down = False

    for i in range(2 * swing_len, n):
        pivot_index = i - swing_len
        # These are per-bar signals in Pine; reset before every bar.
        last_bos_up = False
        last_bos_down = False
        last_choch_up = False
        last_choch_down = False

        # Newly confirmed swing pivots update the active levels first.
        left_highs = highs[pivot_index - swing_len : pivot_index]
        right_highs = highs[pivot_index + 1 : pivot_index + swing_len + 1]
        if left_highs and right_highs and highs[pivot_index] > max(left_highs + right_highs):
            previous = swing_highs[-1] if swing_highs else None
            swing_highs.append(highs[pivot_index])
            if len(swing_highs) > 5:
                swing_highs.pop(0)
            struct_high = highs[pivot_index]
            high_used = False
            if previous is not None:
                high_labels.append(
                    {
                        "type": "high",
                        "label": "HH" if highs[pivot_index] > previous else "LH",
                        "price": round(highs[pivot_index], 2),
                        "time": times[pivot_index],
                    }
                )

        left_lows = lows[pivot_index - swing_len : pivot_index]
        right_lows = lows[pivot_index + 1 : pivot_index + swing_len + 1]
        if left_lows and right_lows and lows[pivot_index] < min(left_lows + right_lows):
            previous = swing_lows[-1] if swing_lows else None
            swing_lows.append(lows[pivot_index])
            if len(swing_lows) > 5:
                swing_lows.pop(0)
            struct_low = lows[pivot_index]
            low_used = False
            if previous is not None:
                low_labels.append(
                    {
                        "type": "low",
                        "label": "HL" if lows[pivot_index] > previous else "LL",
                        "price": round(lows[pivot_index], 2),
                        "time": times[pivot_index],
                    }
                )

        # Breaks are validated on bar close and consumed per active level.
        if struct_high is not None and not high_used and closes[i] > struct_high:
            high_used = True
            if trend == -1:
                last_choch_up = True
                leg_count = 0
                events.append({"type": "CHoCH", "direction": "up", "time": times[i]})
            else:
                last_bos_up = True
                leg_count += 1
                events.append({"type": "BOS", "direction": "up", "time": times[i]})
            trend = 1

        if struct_low is not None and not low_used and closes[i] < struct_low:
            low_used = True
            if trend == 1:
                last_choch_down = True
                leg_count = 0
                events.append({"type": "CHoCH", "direction": "down", "time": times[i]})
            else:
                last_bos_down = True
                leg_count += 1
                events.append({"type": "BOS", "direction": "down", "time": times[i]})
            trend = -1

        if len(events) > max_events:
            events.pop(0)

    health: int | None = None
    state = "NO_STRUCTURE"
    if trend != 0 and len(swing_highs) >= 3 and len(swing_lows) >= 3 and atr_value > 0:
        h1, h2, h3 = swing_highs[-1], swing_highs[-2], swing_highs[-3]
        l1, l2, l3 = swing_lows[-1], swing_lows[-2], swing_lows[-3]
        if trend > 0:
            e1 = (h1 - h2) / atr_value
            e2 = max((h2 - h3) / atr_value, 0.25)
            expansion = _clamp((e1 / e2 + 0.5) / 1.5, 0.0, 1.0) * 100.0
            leg = max(h1 - l2, 1e-9)
            retrace = (h1 - l1) / leg
            pullback = _clamp((1.0 - retrace) / 0.6, 0.0, 1.0) * 100.0
            health = round(0.6 * expansion + 0.4 * pullback)
        else:
            e1 = (l2 - l1) / atr_value
            e2 = max((l3 - l2) / atr_value, 0.25)
            expansion = _clamp((e1 / e2 + 0.5) / 1.5, 0.0, 1.0) * 100.0
            leg = max(h2 - l1, 1e-9)
            retrace = (h1 - l1) / leg
            pullback = _clamp((1.0 - retrace) / 0.6, 0.0, 1.0) * 100.0
            health = round(0.6 * expansion + 0.4 * pullback)
        health = int(_clamp(health, 0, 100))
    if trend != 0:
        new_trend = leg_count < 1
        if new_trend:
            state = "NEW_TREND"
        elif health is None:
            state = "HOLDING"
        elif health < weak_level:
            state = "WEAKENING"
        elif health >= 65:
            state = "STRONG"
        else:
            state = "HOLDING"

    labels = high_labels[-max_events:] + low_labels[-max_events:]
    labels.sort(key=lambda item: item.get("time", ""))
    return {
        "trend": "bullish" if trend > 0 else "bearish" if trend < 0 else "neutral",
        "health": health,
        "health_state": state,
        "bos_up": bool(last_bos_up),
        "bos_down": bool(last_bos_down),
        "choch_up": bool(last_choch_up),
        "choch_down": bool(last_choch_down),
        "swing_high": (
            round(struct_high, 2) if struct_high is not None else None
        ),
        "swing_low": (
            round(struct_low, 2) if struct_low is not None else None
        ),
        "labels": labels[-max_events:],
        "tape": [
            f"{event['type']} {event['direction']}"
            for event in events[-max_events:]
        ],
        "swing_len": int(swing_len),
    }


def _empty_structure(swing_len: int) -> dict[str, Any]:
    return {
        "trend": "neutral",
        "health": None,
        "health_state": "NO_STRUCTURE",
        "bos_up": False,
        "bos_down": False,
        "choch_up": False,
        "choch_down": False,
        "swing_high": None,
        "swing_low": None,
        "labels": [],
        "tape": [],
        "swing_len": int(swing_len),
    }
