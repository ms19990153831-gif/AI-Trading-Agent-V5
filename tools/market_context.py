"""Read-only market context: RSI divergence and tested key levels.

Algorithms are Python ports of the open-source TradingView scripts:
- "RSI Divergence Entry Engine [trade_w_samet]"
- "Support and Resistance Zones, Key Levels & Hold Rate [LunqFX]"

Both readouts are context only. They never open, block or close a trade.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _rsi_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def _atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def _confirmed_pivots(
    df: pd.DataFrame,
    swing_len: int,
) -> list[dict[str, Any]]:
    """Return confirmed pivot bars with their price and kind."""
    highs = [float(v) for v in df["high"]]
    lows = [float(v) for v in df["low"]]
    times = [str(v) for v in df["time"]]
    out: list[dict[str, Any]] = []
    for pivot in range(swing_len, len(df) - swing_len):
        left_highs = highs[pivot - swing_len : pivot]
        right_highs = highs[pivot + 1 : pivot + swing_len + 1]
        if left_highs and right_highs and highs[pivot] > max(left_highs + right_highs):
            out.append(
                {
                    "index": pivot,
                    "price": float(highs[pivot]),
                    "kind": "high",
                    "time": times[pivot],
                }
            )
        left_lows = lows[pivot - swing_len : pivot]
        right_lows = lows[pivot + 1 : pivot + swing_len + 1]
        if left_lows and right_lows and lows[pivot] < min(left_lows + right_lows):
            out.append(
                {
                    "index": pivot,
                    "price": float(lows[pivot]),
                    "kind": "low",
                    "time": times[pivot],
                }
            )
    return out


def momentum_snapshot(
    df: pd.DataFrame,
    rsi_pivot_len: int = 10,
    max_distance: int = 60,
    max_age: int = 20,
) -> dict[str, Any]:
    """Return confirmed regular/hidden RSI divergence as of the last bar."""
    if df is None or len(df) < rsi_pivot_len * 2 + 2:
        return _empty_momentum()
    rsi = _rsi_series(df)
    lows = [float(v) for v in df["low"]]
    highs = [float(v) for v in df["high"]]
    times = [str(v) for v in df["time"]]
    n = len(df)
    low_events: list[dict[str, Any]] = []
    high_events: list[dict[str, Any]] = []
    divergence_events: list[dict[str, Any]] = []

    for pivot in range(rsi_pivot_len, n - rsi_pivot_len):
        r = float(rsi.iloc[pivot])
        if r != r:
            continue
        # Only pivots whose confirmation bar is already closed count.
        confirm = pivot + rsi_pivot_len
        if confirm >= n:
            continue
        left_r = [float(x) for x in rsi.iloc[pivot - rsi_pivot_len : pivot]]
        right_r = [float(x) for x in rsi.iloc[pivot + 1 : pivot + rsi_pivot_len + 1]]
        if left_r and right_r:
            if r < min(left_r + right_r):
                low_events.append(
                    {
                        "pivot": pivot,
                        "confirm": confirm,
                        "rsi": r,
                        "price": float(lows[pivot]),
                        "time": times[pivot],
                    }
                )
            elif r > max(left_r + right_r):
                high_events.append(
                    {
                        "pivot": pivot,
                        "confirm": confirm,
                        "rsi": r,
                        "price": float(highs[pivot]),
                        "time": times[pivot],
                    }
                )

    for events in (low_events, high_events):
        for current_index, current in enumerate(events):
            for previous in reversed(events[:current_index]):
                span = current["confirm"] - previous["confirm"]
                if span < 3:
                    continue
                if span > max_distance:
                    break
                if events is low_events:
                    kind = "bullish"
                    if (
                        current["rsi"] > previous["rsi"]
                        and current["price"] < previous["price"]
                    ):
                        divergence = "regular"
                    elif (
                        current["rsi"] < previous["rsi"]
                        and current["price"] > previous["price"]
                    ):
                        divergence = "hidden"
                    else:
                        continue
                else:
                    kind = "bearish"
                    if (
                        current["rsi"] < previous["rsi"]
                        and current["price"] > previous["price"]
                    ):
                        divergence = "regular"
                    elif (
                        current["rsi"] > previous["rsi"]
                        and current["price"] < previous["price"]
                    ):
                        divergence = "hidden"
                    else:
                        continue
                divergence_events.append(
                    {
                        "type": f"{divergence}_{kind}",
                        "direction": "BUY" if kind == "bullish" else "SELL",
                        "pivot_time": current["time"],
                        "confirm_age": n - 1 - current["confirm"],
                        "price": round(current["price"], 2),
                        "rsi": round(current["rsi"], 1),
                        "previous_rsi": round(previous["rsi"], 1),
                    }
                )
                break

    active_events = [
        event
        for event in divergence_events
        if int(event.get("confirm_age") or 999) <= max_age
    ]
    latest = active_events[-1] if active_events else None
    return {
        "regular_bullish": bool(
            latest and latest["type"] == "regular_bullish"
        ),
        "regular_bearish": bool(
            latest and latest["type"] == "regular_bearish"
        ),
        "hidden_bullish": bool(latest and latest["type"] == "hidden_bullish"),
        "hidden_bearish": bool(latest and latest["type"] == "hidden_bearish"),
        "latest": latest,
        "events": active_events[-3:],
    }


def _empty_momentum() -> dict[str, Any]:
    return {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "latest": None,
        "events": [],
    }


def _cci_series(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mean = tp.rolling(length).mean()
    deviation = (tp - mean).abs().rolling(length).mean()
    return (tp - mean) / (0.015 * deviation.replace(0, float("nan")))


def _price_pivot_near(
    df: pd.DataFrame,
    center: int,
    radius: int,
    strengths: tuple[int, ...],
    kind: str,
) -> bool:
    highs = [float(v) for v in df["high"]]
    lows = [float(v) for v in df["low"]]
    n = len(df)
    for offset in range(-radius, radius + 1):
        index = center + offset
        for strength in strengths:
            if index - strength < 0 or index + strength >= n:
                continue
            if kind == "low":
                left = lows[index - strength : index]
                right = lows[index + 1 : index + strength + 1]
                if left and right and lows[index] < min(left + right):
                    return True
            else:
                left = highs[index - strength : index]
                right = highs[index + 1 : index + strength + 1]
                if left and right and highs[index] > max(left + right):
                    return True
    return False


def cci_divergence_snapshot(
    df: pd.DataFrame,
    cci_length: int = 20,
    extreme_level: float = 150.0,
    max_distance: int = 60,
    max_age: int = 20,
) -> dict[str, Any]:
    """Return confirmed CCI divergence as of the last closed bar."""
    empty = {
        "bullish": False,
        "bearish": False,
        "latest": None,
        "events": [],
    }
    if df is None or len(df) < 40:
        return empty
    cci = _cci_series(df, cci_length)
    lows = [float(v) for v in df["low"]]
    highs = [float(v) for v in df["high"]]
    times = [str(v) for v in df["time"]]
    n = len(df)
    events_by_strength: dict[int, list[dict[str, Any]]] = {}
    strengths = (3, 5, 7, 9)
    for strength in strengths:
        low_events: list[dict[str, Any]] = []
        high_events: list[dict[str, Any]] = []
        for pivot in range(strength, n - strength):
            value = float(cci.iloc[pivot])
            if value != value:
                continue
            confirm = pivot + strength
            if confirm >= n:
                continue
            left = [float(x) for x in cci.iloc[pivot - strength : pivot]]
            right = [float(x) for x in cci.iloc[pivot + 1 : pivot + strength + 1]]
            if value < min(left + right):
                low_events.append(
                    {
                        "pivot": pivot,
                        "confirm": confirm,
                        "cci": value,
                        "price": float(lows[pivot]),
                        "time": times[pivot],
                    }
                )
            elif value > max(left + right):
                high_events.append(
                    {
                        "pivot": pivot,
                        "confirm": confirm,
                        "cci": value,
                        "price": float(highs[pivot]),
                        "time": times[pivot],
                    }
                )

        found: list[dict[str, Any]] = []
        for current in low_events:
            for previous in reversed(low_events[: low_events.index(current)]):
                span = current["confirm"] - previous["confirm"]
                if span > max_distance:
                    break
                if (
                    previous["cci"] <= -extreme_level
                    and current["cci"] > previous["cci"]
                    and current["price"] < previous["price"]
                    and _price_pivot_near(df, current["pivot"], 7, strengths, "low")
                    and _price_pivot_near(df, previous["pivot"], 7, strengths, "low")
                ):
                    found.append(
                        {
                            "type": "bullish",
                            "direction": "BUY",
                            "pivot_time": current["time"],
                            "confirm_age": n - 1 - current["confirm"],
                            "price": round(current["price"], 2),
                            "cci": round(current["cci"], 1),
                            "previous_cci": round(previous["cci"], 1),
                        }
                    )
                    break
        for current in high_events:
            for previous in reversed(high_events[: high_events.index(current)]):
                span = current["confirm"] - previous["confirm"]
                if span > max_distance:
                    break
                if (
                    previous["cci"] >= extreme_level
                    and current["cci"] < previous["cci"]
                    and current["price"] > previous["price"]
                    and _price_pivot_near(df, current["pivot"], 7, strengths, "high")
                    and _price_pivot_near(df, previous["pivot"], 7, strengths, "high")
                ):
                    found.append(
                        {
                            "type": "bearish",
                            "direction": "SELL",
                            "pivot_time": current["time"],
                            "confirm_age": n - 1 - current["confirm"],
                            "price": round(current["price"], 2),
                            "cci": round(current["cci"], 1),
                            "previous_cci": round(previous["cci"], 1),
                        }
                    )
                    break
        events_by_strength[strength] = found

    events = [
        event
        for strength_events in events_by_strength.values()
        for event in strength_events
    ]
    events.sort(key=lambda event: event["pivot_time"])
    deduped: list[dict[str, Any]] = []
    for event in events:
        if (
            deduped
            and event["direction"] == deduped[-1]["direction"]
            and deduped[-1]["pivot_time"] == event["pivot_time"]
        ):
            continue
        deduped.append(event)
    active = [
        event
        for event in deduped
        if int(event.get("confirm_age") or 999) <= max_age
    ]
    latest = active[-1] if active else None
    return {
        "bullish": bool(latest and latest["type"] == "bullish"),
        "bearish": bool(latest and latest["type"] == "bearish"),
        "latest": latest,
        "events": active[-3:],
    }


def cci_reference_text(snapshot: dict[str, Any]) -> str:
    latest = snapshot.get("latest")
    if not latest:
        return "暂无已确认的 CCI 背离"
    direction = "看涨" if latest.get("direction") == "BUY" else "看跌"
    return (
        f"CCI{direction}背离（CCI={latest.get('cci')}，"
        f"前CCI={latest.get('previous_cci')}，价格极值={latest.get('price')}，"
        f"确认后{latest.get('confirm_age')}根K线）"
    )


def ict_context_snapshot(
    df: pd.DataFrame,
    sweep_lookback: int = 20,
    fvg_lookback: int = 30,
) -> dict[str, Any]:
    """Return recent liquidity sweeps and fair-value gaps as read-only context."""
    empty = {
        "sweep": None,
        "fvg": None,
    }
    if df is None or len(df) < max(sweep_lookback, fvg_lookback) + 2:
        return empty
    highs = [float(v) for v in df["high"]]
    lows = [float(v) for v in df["low"]]
    closes = [float(v) for v in df["close"]]
    n = len(df)
    sweep = None
    for bar in range(n - 1, max(sweep_lookback, 1) - 1, -1):
        prior_high = max(highs[bar - sweep_lookback : bar])
        prior_low = min(lows[bar - sweep_lookback : bar])
        if highs[bar] > prior_high and closes[bar] < prior_high:
            sweep = {
                "type": "bearish",
                "direction": "SELL",
                "extreme": round(float(highs[bar]), 2),
                "level": round(float(prior_high), 2),
                "age": n - 1 - bar,
            }
            break
        if lows[bar] < prior_low and closes[bar] > prior_low:
            sweep = {
                "type": "bullish",
                "direction": "BUY",
                "extreme": round(float(lows[bar]), 2),
                "level": round(float(prior_low), 2),
                "age": n - 1 - bar,
            }
            break

    fvg = None
    for bar in range(n - 1, 1, -1):
        if fvg_lookback > 0 and n - 1 - bar > fvg_lookback:
            break
        if lows[bar] > highs[bar - 2]:
            fvg = {
                "type": "bullish",
                "direction": "BUY",
                "top": round(float(lows[bar]), 2),
                "bottom": round(float(highs[bar - 2]), 2),
                "age": n - 1 - bar,
            }
            break
        if highs[bar] < lows[bar - 2]:
            fvg = {
                "type": "bearish",
                "direction": "SELL",
                "top": round(float(lows[bar - 2]), 2),
                "bottom": round(float(highs[bar]), 2),
                "age": n - 1 - bar,
            }
            break
    return {"sweep": sweep, "fvg": fvg}


def ict_reference_text(snapshot: dict[str, Any]) -> str:
    parts = []
    sweep = snapshot.get("sweep")
    if sweep:
        parts.append(
            f"{'向下扫止损' if sweep['type'] == 'bullish' else '向上扫止损'}后收回"
            f"（{sweep['extreme']}，{sweep['age']}根K线前）"
        )
    fvg = snapshot.get("fvg")
    if fvg:
        parts.append(
            f"{'看涨' if fvg['type'] == 'bullish' else '看跌'}FVG "
            f"{fvg['bottom']}-{fvg['top']}（{fvg['age']}根K线前）"
        )
    return "；".join(parts) if parts else "暂无近期流动性扫荡/FVG"


def macd_momentum_snapshot(
    df: pd.DataFrame,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
    trigger_window: int = 5,
) -> dict[str, Any]:
    """Return MACD momentum state and recent confirmed triggers."""
    empty = {
        "state": "neutral",
        "latest_trigger": None,
        "events": [],
        "macd": None,
        "histogram": None,
    }
    if df is None or len(df) < slow_length + signal_length + 2:
        return empty
    close = df["close"]
    macd_series = (
        close.ewm(span=fast_length, adjust=False).mean()
        - close.ewm(span=slow_length, adjust=False).mean()
    )
    signal_series = macd_series.ewm(span=signal_length, adjust=False).mean()
    histogram = macd_series - signal_series
    values = [
        (float(m), float(s), float(h))
        for m, s, h in zip(macd_series, signal_series, histogram)
        if m == m and s == s and h == h
    ]
    if len(values) < 3:
        return empty
    events: list[dict[str, Any]] = []
    start = max(1, len(values) - trigger_window - 1)
    for index in range(len(values) - 1, start - 1, -1):
        macd, signal, hist = values[index]
        prev_macd, prev_signal, prev_hist = values[index - 1]
        prev_hist2 = values[index - 2][2] if index >= 2 else prev_hist
        crossed_up = macd > signal and prev_macd <= prev_signal
        crossed_down = macd < signal and prev_macd >= prev_signal
        if crossed_up:
            trigger = (
                "bullish_reversal_zero"
                if prev_macd < 0
                else "bullish_continuation"
            )
            events.append(
                {
                    "type": trigger,
                    "direction": "BUY",
                    "age": len(values) - 1 - index,
                }
            )
        elif crossed_down:
            trigger = (
                "bearish_reversal_zero"
                if prev_macd > 0
                else "bearish_continuation"
            )
            events.append(
                {
                    "type": trigger,
                    "direction": "SELL",
                    "age": len(values) - 1 - index,
                }
            )
        elif (
            hist > 0
            and hist > prev_hist
            and prev_hist <= prev_hist2
            and macd > signal
        ):
            events.append(
                {"type": "bullish_expansion_turn", "direction": "BUY", "age": len(values) - 1 - index}
            )
        elif (
            hist < 0
            and hist < prev_hist
            and prev_hist >= prev_hist2
            and macd < signal
        ):
            events.append(
                {"type": "bearish_expansion_turn", "direction": "SELL", "age": len(values) - 1 - index}
            )

    latest_macd, latest_signal, latest_hist = values[-1]
    if latest_macd > latest_signal and latest_hist > 0:
        state = "strong_bullish" if latest_hist > values[-2][2] else "weak_bullish"
    elif latest_macd < latest_signal and latest_hist < 0:
        state = "strong_bearish" if latest_hist < values[-2][2] else "weak_bearish"
    else:
        state = "neutral"
    return {
        "state": state,
        "latest_trigger": events[0] if events else None,
        "events": events[:3],
        "macd": round(latest_macd, 4),
        "histogram": round(latest_hist, 4),
    }


def macd_reference_text(snapshot: dict[str, Any]) -> str:
    state_zh = {
        "strong_bullish": "强多头（柱线扩张）",
        "weak_bullish": "多头但柱线收缩",
        "strong_bearish": "强空头（柱线扩张）",
        "weak_bearish": "空头但柱线收缩",
        "neutral": "中性",
    }
    trigger_zh = {
        "bullish_reversal_zero": "零轴下方金叉（反转型）",
        "bullish_continuation": "零轴上方金叉（延续型）",
        "bearish_reversal_zero": "零轴上方死叉（反转型）",
        "bearish_continuation": "零轴下方死叉（延续型）",
        "bullish_expansion_turn": "多头柱线重新扩张",
        "bearish_expansion_turn": "空头柱线重新扩张",
    }
    state = state_zh.get(snapshot.get("state"), "中性")
    latest = snapshot.get("latest_trigger")
    text = f"MACD={state}，柱值={snapshot.get('histogram')}"
    if latest:
        text += (
            f"；近{latest.get('age')}根K线出现"
            f"{trigger_zh.get(str(latest.get('type')), str(latest.get('type')))}"
        )
    return text


def trend_phase_snapshot(indicators: dict[str, Any]) -> dict[str, Any]:
    """Classify the structural phase without making any hard trading decision.

    The output is context for the AI:
    - trend_healthy: structure and momentum still agree
    - exhaustion_risk: trend still exists but reversal risk is rising
    - structure_flip: a CHoCH has already confirmed a change
    """
    structure = indicators.get("structure") or {}
    structure_trend = str(structure.get("trend") or "neutral")
    health_state = str(structure.get("health_state") or "NO_STRUCTURE")
    health = structure.get("health")
    tape = structure.get("tape") or []
    close = float(indicators.get("close") or 0)
    swing_high = structure.get("swing_high")
    swing_low = structure.get("swing_low")
    donchian_high = float(indicators.get("donchian_high") or 0)
    donchian_low = float(indicators.get("donchian_low") or 0)
    atr = float(indicators.get("atr") or 0)
    market_state = str(indicators.get("market_state") or "range")
    rsi = float(indicators.get("rsi") or 50)
    momentum = indicators.get("momentum") or {}
    cci = indicators.get("cci") or {}
    ict = indicators.get("ict") or {}

    if structure_trend == "neutral" or (swing_high is None and swing_low is None):
        return {"phase": "neutral", "reason": "无明确结构阶段"}

    latest_tape = str(tape[-1]) if tape else ""
    reasons: list[str] = []
    phase = "trend_healthy"

    if "CHoCH" in latest_tape:
        phase = "structure_flip"
        reasons.append("近期已出现CHoCH结构翻转确认")
    else:
        if health_state == "WEAKENING":
            phase = "exhaustion_risk"
            reasons.append(f"趋势健康度{health}/100转弱")
        if momentum.get("latest"):
            direction = str(momentum["latest"].get("direction", "")).upper()
            age = int(momentum["latest"].get("confirm_age") or 999)
            if (
                structure_trend == "bearish"
                and direction == "BUY"
                and age <= 30
            ):
                phase = "exhaustion_risk"
                reasons.append(f"RSI看涨背离确认后{age}根K线")
            elif (
                structure_trend == "bullish"
                and direction == "SELL"
                and age <= 30
            ):
                phase = "exhaustion_risk"
                reasons.append(f"RSI看跌背离确认后{age}根K线")
        if cci.get("latest"):
            direction = str(cci["latest"].get("direction", "")).upper()
            age = int(cci["latest"].get("confirm_age") or 999)
            if (
                structure_trend == "bearish"
                and direction == "BUY"
                and age <= 30
            ):
                phase = "exhaustion_risk"
                reasons.append(f"CCI看涨背离确认后{age}根K线")
            elif (
                structure_trend == "bullish"
                and direction == "SELL"
                and age <= 30
            ):
                phase = "exhaustion_risk"
                reasons.append(f"CCI看跌背离确认后{age}根K线")
        if (
            swing_high is not None
            and swing_low is not None
            and float(swing_high) > float(swing_low)
            and atr > 0
        ):
            low = float(swing_low)
            high = float(swing_high)
            position = (close - low) / (high - low)
            if structure_trend == "bearish" and position >= 0.55:
                reasons.append(
                    f"空头结构中价格已回撤到区间{position:.0%}"
                )
                phase = "exhaustion_risk"
            elif structure_trend == "bullish" and position <= 0.45:
                reasons.append(
                    f"多头结构中价格回落到区间{position:.0%}"
                )
                phase = "exhaustion_risk"
        if (
            structure_trend == "bearish"
            and market_state == "trend"
            and rsi <= 30
            and donchian_low > 0
            and close <= donchian_low + 0.8 * atr
        ):
            phase = "exhaustion_risk"
            reasons.append("价格贴近近期低点且RSI深度超卖")
        elif (
            structure_trend == "bullish"
            and market_state == "trend"
            and rsi >= 70
            and donchian_high > 0
            and close >= donchian_high - 0.8 * atr
        ):
            phase = "exhaustion_risk"
            reasons.append("价格贴近近期高点且RSI深度超买")

    if phase == "trend_healthy" and health_state == "HOLDING":
        reasons.append("结构完整但健康度一般，只做同向确认入场")
    elif phase == "trend_healthy" and health_state == "STRONG":
        reasons.append("结构完整且健康度强")
    return {
        "phase": phase,
        "trend": structure_trend,
        "reason": "；".join(reasons) if reasons else "结构健康，等待同向确认",
    }


def trend_phase_reference_text(phase: dict[str, Any]) -> str:
    phase_zh = {
        "trend_healthy": "结构健康",
        "exhaustion_risk": "末端风险",
        "structure_flip": "结构翻转",
        "neutral": "中性",
    }
    trend_zh = {"bullish": "多头", "bearish": "空头"}
    name = phase_zh.get(str(phase.get("phase")), str(phase.get("phase")))
    trend = trend_zh.get(str(phase.get("trend")), str(phase.get("trend")))
    return f"{trend}结构阶段={name}（{phase.get('reason')}）"


def key_level_snapshot(
    df: pd.DataFrame,
    swing_len: int = 10,
    zone_atr: float = 0.75,
    merge_atr: float = 1.0,
    max_wide_atr: float = 1.6,
    rejection_atr: float = 0.2,
    test_cool_bars: int = 8,
    max_zones: int = 8,
) -> dict[str, Any]:
    """Return tested key zones as support/resistance context.

    Zones are built from confirmed swing pivots and merge when pivots sit
    within `merge_atr` x ATR. Test/hold counts follow the source's close-based
    rejection rules: a hold requires the close to clear the zone by
    `rejection_atr` x ATR on the side it came from.
    """
    empty = {
        "support": None,
        "resistance": None,
        "zones": [],
    }
    if df is None or len(df) < swing_len * 2 + 4:
        return empty
    highs = [float(v) for v in df["high"]]
    lows = [float(v) for v in df["low"]]
    closes = [float(v) for v in df["close"]]
    times = [str(v) for v in df["time"]]
    n = len(df)
    atr_series = _atr_series(df)

    zones: list[dict[str, Any]] = []

    def add_zone(price: float, confirm: int) -> None:
        if confirm >= n:
            return
        atr = float(atr_series.iloc[confirm])
        if atr != atr or atr <= 0:
            atr = 1.0
        half = atr * zone_atr / 2.0
        tol = atr * merge_atr
        top = price + half
        bottom = price - half
        best = -1
        best_distance = float("inf")
        for index, zone in enumerate(zones):
            mid = (zone["top"] + zone["bottom"]) / 2.0
            distance = abs(mid - price)
            if distance <= tol and distance < best_distance:
                best = index
                best_distance = distance
        if best >= 0:
            zone = zones[best]
            zone["top"] = max(zone["top"], top)
            zone["bottom"] = min(zone["bottom"], bottom)
            cap = atr * max_wide_atr
            if zone["top"] - zone["bottom"] > cap:
                zone["top"] = price + cap / 2.0
                zone["bottom"] = price - cap / 2.0
        else:
            zones.append(
                {
                    "top": top,
                    "bottom": bottom,
                    "tests": 0,
                    "holds": 0,
                    "state": 0,
                    "last": -10000,
                    "birth": confirm,
                    "price": price,
                }
            )
        while len(zones) > max_zones:
            zones.pop(0)

    for pivot in _confirmed_pivots(df, swing_len):
        add_zone(pivot["price"], pivot["index"] + swing_len)

    for bar in range(1, n):
        high = highs[bar]
        low = lows[bar]
        close = closes[bar]
        prev_close = closes[bar - 1]
        for zone in zones:
            if bar <= zone["birth"]:
                continue
            atr = float(atr_series.iloc[bar])
            if atr != atr or atr <= 0:
                atr = 1.0
            top = zone["top"]
            bottom = zone["bottom"]
            inside = high >= bottom and low <= top
            if zone["state"] == 0 and inside and bar - zone["last"] >= test_cool_bars:
                if prev_close < bottom:
                    zone["state"] = 1
                    zone["tests"] += 1
                elif prev_close > top:
                    zone["state"] = 2
                    zone["tests"] += 1
            elif zone["state"] == 1:
                if close > top:
                    zone["state"] = 0
                    zone["last"] = bar
                elif close < bottom - atr * rejection_atr:
                    zone["holds"] += 1
                    zone["state"] = 0
                    zone["last"] = bar
            elif zone["state"] == 2:
                if close < bottom:
                    zone["state"] = 0
                    zone["last"] = bar
                elif close > top + atr * rejection_atr:
                    zone["holds"] += 1
                    zone["state"] = 0
                    zone["last"] = bar

    last_close = float(closes[-1])
    current_atr = float(atr_series.iloc[-1])
    if current_atr != current_atr or current_atr <= 0:
        current_atr = 1.0
    rows: list[dict[str, Any]] = []
    for zone in zones:
        mid = (zone["top"] + zone["bottom"]) / 2.0
        tests = int(zone["tests"])
        if tests <= 0:
            continue
        holds = int(zone["holds"])
        rows.append(
            {
                "top": round(zone["top"], 2),
                "bottom": round(zone["bottom"], 2),
                "mid": round(mid, 2),
                "tests": tests,
                "holds": holds,
                "hold_rate": (
                    round(holds / tests * 100.0, 1) if tests > 0 else 0.0
                ),
                "distance_atr": round((mid - last_close) / current_atr, 2),
            }
        )
    rows.sort(key=lambda row: abs(row["mid"] - last_close))
    resistance = next((r for r in rows if r["mid"] > last_close), None)
    support = next((r for r in rows if r["mid"] < last_close), None)
    return {
        "support": support,
        "resistance": resistance,
        "zones": rows[:4],
    }


def key_level_reference_text(snapshot: dict[str, Any]) -> str:
    parts = []
    resistance = snapshot.get("resistance")
    support = snapshot.get("support")
    if support:
        parts.append(
            f"支撑{support['mid']}（测试{support['tests']}次，"
            f"守住{support['holds']}次，{support['hold_rate']}%，"
            f"距当前{support['distance_atr']}×ATR）"
        )
    if resistance:
        parts.append(
            f"阻力{resistance['mid']}（测试{resistance['tests']}次，"
            f"守住{resistance['holds']}次，{resistance['hold_rate']}%，"
            f"距当前{resistance['distance_atr']}×ATR）"
        )
    if not parts:
        return "暂无已测试的关键支撑/阻力"
    return "；".join(parts)


def momentum_reference_text(snapshot: dict[str, Any]) -> str:
    latest = snapshot.get("latest")
    if not latest:
        return "暂无已确认的动量背离"
    direction = "看涨" if latest.get("direction") == "BUY" else "看跌"
    divergence_type = (
        "常规"
        if str(latest.get("type", "")).startswith("regular")
        else "隐藏"
    )
    return (
        f"{divergence_type}{direction}背离（价格极值={latest.get('price')}，"
        f"RSI={latest.get('rsi')}，前RSI={latest.get('previous_rsi')}，"
        f"确认后{latest.get('confirm_age')}根K线）"
    )
