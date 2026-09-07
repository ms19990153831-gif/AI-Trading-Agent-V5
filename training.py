"""Historical training and self-evolution module (Part 8 style).

Simulates deterministic strategy personas across many historical environments,
scores them, and stores the best persona into trading memory.
"""

from __future__ import annotations

import random

from config import INITIAL_BALANCE, MAX_RISK_PERCENT, SIMULATE, SYMBOL, TIMEFRAME
from memory.database import Database
from mt5.market import MT5MarketData, SimulatedFeed, compute_indicators


def _signal(action: str, confidence: int, reason: str) -> dict:
    return {
        "action": action,
        "confidence": confidence,
        "reason": reason,
        "risk_percent": MAX_RISK_PERCENT,
    }


def trend_follow(ind: dict) -> dict:
    if ind["trend"] == "bullish" and ind["rsi"] < 70:
        return _signal("BUY", 75, "trend follow long")
    if ind["trend"] == "bearish" and ind["rsi"] > 30:
        return _signal("SELL", 75, "trend follow short")
    return _signal("WAIT", 40, "no trend edge")


def mean_reversion(ind: dict) -> dict:
    if ind["rsi"] < 30:
        return _signal("BUY", 72, "oversold mean reversion")
    if ind["rsi"] > 70:
        return _signal("SELL", 72, "overbought mean reversion")
    return _signal("WAIT", 40, "rsi neutral")


def momentum(ind: dict) -> dict:
    rsi = ind["rsi"]
    if ind["trend"] == "bullish" and 50 < rsi < 75:
        return _signal("BUY", 74, "bullish momentum")
    if ind["trend"] == "bearish" and 25 < rsi < 50:
        return _signal("SELL", 74, "bearish momentum")
    return _signal("WAIT", 40, "momentum flat")


def breakout(ind: dict) -> dict:
    if ind["trend"] == "bullish":
        return _signal("BUY", 76, "breakout long")
    if ind["trend"] == "bearish":
        return _signal("SELL", 76, "breakout short")
    return _signal("WAIT", 40, "range")


def conservative(ind: dict) -> dict:
    if ind["trend"] == "bullish" and 35 < ind["rsi"] < 60:
        return _signal("BUY", 80, "conservative long")
    if ind["trend"] == "bearish" and 40 < ind["rsi"] < 65:
        return _signal("SELL", 80, "conservative short")
    return _signal("WAIT", 40, "wait for cleaner setup")


PERSONAS = {
    "trend_follow": trend_follow,
    "mean_reversion": mean_reversion,
    "momentum": momentum,
    "breakout": breakout,
    "conservative": conservative,
}


def _simulate_env(df, signal_fn, warmup: int = 60):
    balance = INITIAL_BALANCE
    positions: list[dict] = []
    closed: list[dict] = []
    equity: list[float] = []

    for i in range(warmup, len(df)):
        bar = df.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        for pos in list(positions):
            side = pos["side"]
            entry = pos["entry"]
            sl = pos["sl"]
            tp = pos["tp"]
            vol = pos["volume"]
            if side == "BUY":
                if low <= sl:
                    pnl = (sl - entry) * vol * 100
                    closed.append({"pnl": pnl})
                    balance += pnl
                    positions.remove(pos)
                    continue
                if high >= tp:
                    pnl = (tp - entry) * vol * 100
                    closed.append({"pnl": pnl})
                    balance += pnl
                    positions.remove(pos)
                    continue
            else:
                if high >= sl:
                    pnl = (entry - sl) * vol * 100
                    closed.append({"pnl": pnl})
                    balance += pnl
                    positions.remove(pos)
                    continue
                if low <= tp:
                    pnl = (entry - tp) * vol * 100
                    closed.append({"pnl": pnl})
                    balance += pnl
                    positions.remove(pos)

        window = df.iloc[: i + 1].tail(warmup)
        ind = compute_indicators(window)
        signal = signal_fn(ind)
        if signal["action"] != "WAIT" and len(positions) < 3:
            distance = max(float(ind["atr"]) * 1.5, float(ind["close"]) * 0.001)
            volume = max(0.01, round((balance * MAX_RISK_PERCENT) / (distance * 100), 2))
            entry = float(ind["close"])
            sl = entry - distance if signal["action"] == "BUY" else entry + distance
            tp = entry + distance * 2 if signal["action"] == "BUY" else entry - distance * 2
            positions.append(
                {
                    "side": signal["action"],
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "volume": volume,
                }
            )

        mark = sum(
            (close - pos["entry"]) * pos["volume"] * 100
            if pos["side"] == "BUY"
            else (pos["entry"] - close) * pos["volume"] * 100
            for pos in positions
        )
        equity.append(balance + mark)

    if positions and len(df):
        last_close = float(df.iloc[-1]["close"])
        for pos in positions:
            pnl = (
                (last_close - pos["entry"]) * pos["volume"] * 100
                if pos["side"] == "BUY"
                else (pos["entry"] - last_close) * pos["volume"] * 100
            )
            closed.append({"pnl": pnl})
            balance += pnl
    return closed, equity


def _max_drawdown(equity: list[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def run_training(
    db: Database,
    environments: int = 20,
    bars: int = 2500,
    seed: int = 7,
) -> list[dict]:
    if SIMULATE:
        df = SimulatedFeed().fetch_bars(bars, TIMEFRAME)
    else:
        df = MT5MarketData().get_bars(SYMBOL, TIMEFRAME, bars)
    if len(df) < 400:
        raise RuntimeError("not enough history bars for training")

    rng = random.Random(seed)
    results: list[dict] = []
    for name, signal_fn in PERSONAS.items():
        pnls: list[float] = []
        worst_drawdown = 0.0
        env_count = 0
        for _ in range(environments):
            start = rng.randint(0, len(df) - 400)
            env = df.iloc[start : start + 400].reset_index(drop=True)
            closed, equity = _simulate_env(env, signal_fn)
            if not equity:
                continue
            env_count += 1
            pnls.extend(trade["pnl"] for trade in closed)
            worst_drawdown = max(worst_drawdown, _max_drawdown(equity))

        wins = sum(1 for pnl in pnls if pnl > 0)
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = -sum(pnl for pnl in pnls if pnl < 0)
        net = sum(pnls)
        metrics = {
            "trades": len(pnls),
            "wins": wins,
            "win_rate": round(wins / max(len(pnls), 1), 4),
            "profit_factor": round(gross_profit / max(gross_loss, 1e-9), 3),
            "max_drawdown": round(worst_drawdown, 4),
            "net_pnl": round(net, 2),
        }
        score = round(net - worst_drawdown * INITIAL_BALANCE + wins * 2, 2)
        db.save_training_run(name, env_count, metrics, score)
        results.append({"persona": name, "score": score, **metrics})

    results.sort(key=lambda item: item["score"], reverse=True)
    if results:
        best = results[0]
        db.save_experience(
            f"best training persona: {best['persona']} score={best['score']} "
            f"net_pnl={best['net_pnl']} win_rate={best['win_rate']}",
            "training",
        )
    return results
