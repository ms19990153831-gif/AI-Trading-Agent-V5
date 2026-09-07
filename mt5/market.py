"""Market data: MT5 feed or a deterministic simulated feed."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from config import (
    BAR_COUNT,
    BOX_BOUNDARY_ATR,
    CCI_DIVERGENCE_EXTREME,
    CCI_DIVERGENCE_MAX_DISTANCE,
    CCI_LENGTH,
    CONTEXT_CLOSES,
    DIVERGENCE_VALID_BARS,
    ICT_FVG_LOOKBACK,
    ICT_SWEEP_LOOKBACK,
    KEY_ZONE_ATR,
    KEY_ZONE_SWING_LEN,
    RSI_DIVERGENCE_MAX_DISTANCE,
    RSI_DIVERGENCE_PIVOT_LEN,
    SIMULATE,
    SIM_SEED,
    SIM_VOLATILITY,
    STRUCTURE_SWING_LEN,
    SYMBOL,
    TIMEFRAME,
    TREND_HEALTH_WEAK_LEVEL,
)
from tools.market_context import (
    cci_divergence_snapshot,
    cci_reference_text,
    ict_context_snapshot,
    ict_reference_text,
    key_level_reference_text,
    key_level_snapshot,
    macd_momentum_snapshot,
    macd_reference_text,
    momentum_reference_text,
    momentum_snapshot,
    trend_phase_reference_text,
    trend_phase_snapshot,
)
from tools.market_structure import structure_snapshot

TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}


class SimulatedFeed:
    """Regime-switching random walk used when MT5 is not available."""

    def __init__(
        self,
        symbol: str = SYMBOL,
        seed: int = SIM_SEED,
        volatility: float = SIM_VOLATILITY,
    ) -> None:
        self.symbol = symbol
        self.rng = random.Random(seed)
        self.volatility = volatility
        self.price = 3350.0
        self.cursor = 0
        self.t0 = datetime(2026, 1, 1, 0, 0)
        self._preheat(240)

    def _bar(self) -> tuple[float, float, float, float]:
        regime = self.rng.choices(["bull", "bear", "range"], weights=[0.42, 0.33, 0.25])[0]
        drift = 0.18 if regime == "bull" else -0.18 if regime == "bear" else 0.0
        open_price = self.price
        close = open_price + drift + self.rng.gauss(0, self.volatility)
        high = max(open_price, close) + abs(self.rng.gauss(0, self.volatility * 0.45))
        low = min(open_price, close) - abs(self.rng.gauss(0, self.volatility * 0.45))
        self.price = close
        return open_price, high, low, close

    def _preheat(self, count: int) -> None:
        for _ in range(count):
            self._bar()

    def fetch_bars(self, count: int = BAR_COUNT, timeframe: str = TIMEFRAME) -> pd.DataFrame:
        step = TIMEFRAME_SECONDS.get(timeframe, 3600)
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            open_price, high, low, close = self._bar()
            ts = self.t0 + timedelta(seconds=self.cursor * step)
            self.cursor += 1
            rows.append(
                {
                    "time": ts,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "tick_volume": int(50 + self.rng.random() * 500),
                }
            )
        return pd.DataFrame(rows)


class MT5MarketData:
    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        import MetaTrader5 as mt5

        tf = getattr(mt5, f"TIMEFRAME_{timeframe}", mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 returned no data for {symbol}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df


class MarketData:
    def __init__(self) -> None:
        self.feed = SimulatedFeed() if SIMULATE else None
        self.mt5 = None if SIMULATE else MT5MarketData()

    def get_bars(
        self,
        symbol: str = SYMBOL,
        timeframe: str = TIMEFRAME,
        count: int = BAR_COUNT,
    ) -> pd.DataFrame:
        if SIMULATE:
            return self.feed.fetch_bars(count, timeframe)
        return self.mt5.get_bars(symbol, timeframe, count)


def compute_indicators(
    df: pd.DataFrame,
    htf: dict | str | None = None,
) -> dict[str, Any]:
    """Compute the numeric context consumed by agents and mock brain."""
    if isinstance(htf, str):
        htf = {"trend": htf}
    htf = htf or {}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        [v if (v > d and v > 0) else 0.0 for v, d in zip(up_move, down_move)],
        index=df.index,
    )
    minus_dm = pd.Series(
        [d if (d > v and d > 0) else 0.0 for v, d in zip(up_move, down_move)],
        index=df.index,
    )
    tr_smooth = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx = float(dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    if adx != adx:
        adx = 0.0

    last_close = float(close.iloc[-1])
    trend = "bullish" if float(sma20.iloc[-1]) > float(sma50.iloc[-1]) else "bearish"
    sma20_val = float(sma20.iloc[-1])
    sma50_val = float(sma50.iloc[-1])
    sma_aligned = (
        (last_close > sma20_val > sma50_val)
        or (last_close < sma20_val < sma50_val)
    )
    sma_gap = abs(sma20_val - sma50_val) / max(float(atr) if atr == atr else 1.0, 1e-9)
    atr20 = float(true_range.rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(atr)
    donchian_high = float(high.tail(20).max()) if len(df) >= 20 else float(high.max())
    donchian_low = float(low.tail(20).min()) if len(df) >= 20 else float(low.min())
    near_extreme = last_close >= donchian_high * 0.995 or last_close <= donchian_low * 1.005

    last_low = float(low.iloc[-1])
    last_high = float(high.iloc[-1])
    two_b_bottom = False
    two_b_top = False
    if len(df) >= 22:
        ref_low = float(low.iloc[-22:-2].min())
        ref_high = float(high.iloc[-22:-2].max())
        prev_low = float(low.iloc[-2])
        prev_high = float(high.iloc[-2])
        two_b_bottom = (
            prev_low < ref_low
            and last_low >= prev_low
            and last_close > ref_low
        )
        two_b_top = (
            prev_high > ref_high
            and last_high <= prev_high
            and last_close < ref_high
        )
    elif len(df) >= 3:
        prev_low = float(low.iloc[-2])
        prev_high = float(high.iloc[-2])
        ref_low = float(low.iloc[:-2].min())
        ref_high = float(high.iloc[:-2].max())
        two_b_bottom = (
            prev_low < ref_low
            and last_low >= prev_low
            and last_close > ref_low
        )
        two_b_top = (
            prev_high > ref_high
            and last_high <= prev_high
            and last_close < ref_high
        )
    elif len(df) >= 2:
        prev_low = float(low.iloc[-2])
        prev_high = float(high.iloc[-2])
        two_b_bottom = last_low < donchian_low and last_close > donchian_low
        two_b_top = last_high > donchian_high and last_close < donchian_high
    boundary_atr = float(BOX_BOUNDARY_ATR)
    box_zone_bottom = round(donchian_low + boundary_atr * float(atr), 2)
    box_zone_top = round(donchian_high - boundary_atr * float(atr), 2)

    state_score = 0
    if adx >= 25:
        state_score += 1
    if sma_aligned:
        state_score += 1
    if sma_gap >= 1.2:
        state_score += 1
    if atr and atr20 and atr > atr20 * 1.15:
        state_score += 1
    if near_extreme:
        state_score += 1
    market_state = "trend" if state_score >= 3 else "range"
    state_confidence = (
        "high" if state_score >= 4 else "medium" if state_score == 3 else "low"
    )
    momentum_ok = (
        (trend == "bullish" and 50 <= float(rsi) <= 78)
        or (trend == "bearish" and 22 <= float(rsi) <= 50)
    )
    range_signal = "none"
    if market_state == "range":
        if float(rsi) >= 70 or last_close >= donchian_high * 0.998:
            range_signal = "SELL"
        elif float(rsi) <= 30 or last_close <= donchian_low * 1.002:
            range_signal = "BUY"
    structure = structure_snapshot(
        df,
        swing_len=int(STRUCTURE_SWING_LEN),
        weak_level=int(TREND_HEALTH_WEAK_LEVEL),
    )
    momentum = momentum_snapshot(
        df,
        rsi_pivot_len=int(RSI_DIVERGENCE_PIVOT_LEN),
        max_distance=int(RSI_DIVERGENCE_MAX_DISTANCE),
        max_age=int(DIVERGENCE_VALID_BARS),
    )
    key_levels = key_level_snapshot(
        df,
        swing_len=int(KEY_ZONE_SWING_LEN),
        zone_atr=float(KEY_ZONE_ATR),
    )
    cci = cci_divergence_snapshot(
        df,
        cci_length=int(CCI_LENGTH),
        extreme_level=float(CCI_DIVERGENCE_EXTREME),
        max_distance=int(CCI_DIVERGENCE_MAX_DISTANCE),
        max_age=int(DIVERGENCE_VALID_BARS),
    )
    ict = ict_context_snapshot(
        df,
        sweep_lookback=int(ICT_SWEEP_LOOKBACK),
        fvg_lookback=int(ICT_FVG_LOOKBACK),
    )
    macd = macd_momentum_snapshot(df)
    output = {
        "symbol": SYMBOL,
        "close": round(last_close, 2),
        "sma20": round(float(sma20.iloc[-1]), 2),
        "sma50": round(float(sma50.iloc[-1]), 2),
        "rsi": round(float(rsi), 1) if rsi == rsi else 50.0,
        "atr": round(float(atr), 2) if atr == atr else 1.0,
        "adx": round(adx, 1),
        "market_state": market_state,
        "momentum_ok": momentum_ok,
        "range_signal": range_signal,
        "state_score": state_score,
        "state_confidence": state_confidence,
        "trend": trend,
        "htf_trend": str(htf.get("trend") or "neutral"),
        "htf_rsi": round(float(htf.get("rsi") or 50), 1),
        "htf_adx": round(float(htf.get("adx") or 0), 1),
        "volatility": "high" if atr and atr > last_close * 0.002 else "medium",
        "last_high": round(float(high.iloc[-1]), 2),
        "last_low": round(float(low.iloc[-1]), 2),
        "donchian_high": round(donchian_high, 2),
        "donchian_low": round(donchian_low, 2),
        "near_extreme": bool(near_extreme),
        "box_zone_bottom": box_zone_bottom,
        "box_zone_top": box_zone_top,
        "two_b_bottom": bool(two_b_bottom),
        "two_b_top": bool(two_b_top),
        "structure": structure,
        "momentum": momentum,
        "key_levels": key_levels,
        "momentum_reference": momentum_reference_text(momentum),
        "key_level_reference": key_level_reference_text(key_levels),
        "cci": cci,
        "cci_reference": cci_reference_text(cci),
        "ict": ict,
        "ict_reference": ict_reference_text(ict),
        "macd": macd,
        "macd_reference": macd_reference_text(macd),
        "closes": [round(float(x), 2) for x in close.tail(CONTEXT_CLOSES).tolist()],
    }
    trend_phase = trend_phase_snapshot(output)
    output["trend_phase"] = trend_phase
    output["trend_phase_reference"] = trend_phase_reference_text(trend_phase)
    return output


def htf_trend_from_df(df: pd.DataFrame) -> str | None:
    """Higher-timeframe trend from SMA20/SMA50, used to filter M15 entries."""
    if df is None or len(df) < 50:
        return None
    sma20 = float(df["close"].rolling(20).mean().iloc[-1])
    sma50 = float(df["close"].rolling(50).mean().iloc[-1])
    if sma20 > sma50:
        return "bullish"
    if sma20 < sma50:
        return "bearish"
    return "neutral"


def htf_snapshot_from_df(df: pd.DataFrame) -> dict:
    """H1 confluence snapshot: trend, RSI and ADX for the last bar."""
    if df is None or len(df) < 60:
        return {}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    if sma20.iloc[-1] > sma50.iloc[-1]:
        trend = "bullish"
    elif sma20.iloc[-1] < sma50.iloc[-1]:
        trend = "bearish"
    else:
        trend = "neutral"
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    if rsi != rsi:
        rsi = 50.0
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        [v if (v > d and v > 0) else 0.0 for v, d in zip(up_move, down_move)],
        index=df.index,
    )
    minus_dm = pd.Series(
        [d if (d > v and d > 0) else 0.0 for v, d in zip(up_move, down_move)],
        index=df.index,
    )
    tr_smooth = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx = float(dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    if adx != adx:
        adx = 0.0
    return {"trend": trend, "rsi": round(rsi, 1), "adx": round(adx, 1)}


def build_context(df: pd.DataFrame, indicators: dict[str, Any]) -> dict[str, Any]:
    latest = df.tail(1).to_dict("records")
    for row in latest:
        row["time"] = str(row["time"])
    return {
        "indicators": indicators,
        "last_bars": latest,
        "bars_count": len(df),
    }
