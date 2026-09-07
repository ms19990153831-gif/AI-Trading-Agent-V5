"""Historical backtest: replay bars through the real agent pipeline.

Uses simulated bar timestamps so results can be grouped by year, covering both
bull and bear regimes instead of a single market state.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent.trader_agent import TraderAgent
from agent.strategy_advisor import StrategyAdvisor
from brain.llm import MockLLM
from config import (
    BAR_COUNT,
    DATA_DIR,
    INITIAL_BALANCE,
    MAX_LOSS_STREAK,
    MAX_RISK_PERCENT,
    SIMULATE,
    SYMBOL,
    TIMEFRAME,
)
from execution.paper_broker import PaperBroker
from memory.database import Database
from mt5.market import MT5MarketData, SimulatedFeed, compute_indicators
from risk.manager import RiskManager
from tools.calculator import price_precision, symbol_spread
from vision.analyzer import VisionAnalyzer


def _load_history(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """Prefer a full CSV history when present, otherwise use MT5/simulation."""
    csv_path = os.environ.get(
        "BACKTEST_HISTORY_CSV",
        os.path.join(DATA_DIR, f"history_{symbol}_{timeframe}.csv"),
    )
    if os.path.exists(csv_path):
        df = pd.read_csv(
            csv_path,
            parse_dates=["time"],
            on_bad_lines="skip",
        )
        return df.tail(bars).reset_index(drop=True)
    if SIMULATE:
        return SimulatedFeed().fetch_bars(bars, timeframe)
    return MT5MarketData().get_bars(symbol, timeframe, bars)


def _build_htf_context(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Build completed H1 trend/ADX/RSI by resampling the same M15 feed."""
    if df_m15 is None or len(df_m15) < 240:
        return pd.DataFrame(columns=["time", "htf_trend", "htf_adx", "htf_rsi"])
    frame = df_m15.sort_values("time").set_index("time")
    h1 = (
        frame.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open", "close"])
        .reset_index()
    )
    close = h1["close"]
    high = h1["high"]
    low = h1["low"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    trend = pd.Series("neutral", index=h1.index)
    trend[sma20 > sma50] = "bullish"
    trend[sma20 < sma50] = "bearish"
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        [v if (v > d and v > 0) else 0.0 for v, d in zip(up_move, down_move)],
        index=h1.index,
    )
    minus_dm = pd.Series(
        [d if (d > v and d > 0) else 0.0 for v, d in zip(up_move, down_move)],
        index=h1.index,
    )
    tr_smooth = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth.replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    result = h1.assign(
        htf_trend=trend,
        htf_adx=adx,
        htf_rsi=rsi,
    )[["time", "htf_trend", "htf_adx", "htf_rsi"]].dropna()
    # Use only completed H1 bars: shift the bar time by one hour so a M15
    # signal never sees the close of the H1 bar that is still forming.
    result["time"] = result["time"] + pd.Timedelta(hours=1)
    return result


class _BacktestAccount:
    def __init__(self, initial: float) -> None:
        self.balance = initial
        self.open_positions: dict[int, dict] = {}
        self.daily_pnl = 0.0
        self.loss_streak = 0
        self.last_day: str | None = None
        self.last_loss_ts: str | None = None
        self.now_ts = ""
        self.last_price = 0.0

    def snapshot(self) -> dict:
        paused = False
        streak = self.loss_streak
        if self.last_loss_ts and streak >= MAX_LOSS_STREAK:
            try:
                last = datetime.fromisoformat(self.last_loss_ts)
                now = datetime.fromisoformat(self.now_ts)
                paused = (now - last).total_seconds() < 24 * 3600
            except ValueError:
                paused = True
            if not paused:
                streak = 0
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.balance, 2),
            "open_positions": len(self.open_positions),
            "daily_pnl": round(self.daily_pnl, 2),
            "loss_streak": streak,
            "loss_paused": paused,
            "positions": [
                {
                    "side": pos["side"],
                    "volume": float(pos["volume"]),
                    "entry": float(pos["entry"]),
                    "profit": round(
                        (
                            self.last_price - float(pos["entry"])
                        )
                        * (1.0 if pos["side"] == "BUY" else -1.0)
                        * float(pos["volume"])
                        * 100,
                        2,
                    )
                    if self.last_price
                    else 0.0,
                }
                for pos in self.open_positions.values()
            ],
        }

    def on_open(self, trade: dict) -> None:
        self.open_positions[trade["id"]] = {
            "side": trade["side"],
            "entry": float(trade["entry"]),
            "volume": float(trade["volume"]),
        }

    def on_events(self, day: str, ts: str, events: list[dict]) -> None:
        self.now_ts = ts
        if self.last_day != day:
            self.last_day = day
            self.daily_pnl = 0.0
        for event in events:
            if event["type"] not in ("CLOSE_SL", "CLOSE_TP", "CLOSE_END"):
                continue
            pnl = float(event.get("pnl") or 0.0)
            self.balance += pnl
            self.daily_pnl += pnl
            self.open_positions.pop(event.get("trade_id"), None)
            if pnl < 0:
                self.loss_streak += 1
                if self.loss_streak >= MAX_LOSS_STREAK:
                    self.last_loss_ts = ts
            else:
                self.loss_streak = 0

    def equity(self, price: float) -> float:
        mark = 0.0
        for pos in self.open_positions.values():
            direction = 1.0 if pos["side"] == "BUY" else -1.0
            mark += (price - pos["entry"]) * direction * pos["volume"] * 100
        return self.balance + mark


def _metrics(closed: list[dict], equity: list[float]) -> dict:
    pnls = [float(trade["pnl"] or 0.0) for trade in closed]
    wins = [pnl for pnl in pnls if pnl > 0]
    gross_profit = sum(wins)
    gross_loss = -sum(pnl for pnl in pnls if pnl < 0)
    peak = float("-inf")
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    return {
        "closed_trades": len(closed),
        "wins": len(wins),
        "win_rate": round(len(wins) / max(len(closed), 1), 4),
        "profit_factor": round(gross_profit / max(gross_loss, 1e-9), 3),
        "max_drawdown": round(max_drawdown, 4),
        "net_pnl": round(sum(pnls), 2),
        "final_equity": round(equity[-1], 2) if equity else INITIAL_BALANCE,
    }


def _year_stats(closed: list[dict]) -> dict:
    by_year: dict[str, dict] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "net_pnl": 0.0}
    )
    for trade in closed:
        year = (trade.get("exit_ts") or "")[:4]
        if not year:
            continue
        stats = by_year[year]
        stats["trades"] += 1
        pnl = float(trade.get("pnl") or 0.0)
        stats["net_pnl"] += pnl
        if pnl > 0:
            stats["wins"] += 1
    output = {}
    for year, stats in sorted(by_year.items()):
        output[year] = {
            "trades": stats["trades"],
            "wins": stats["wins"],
            "win_rate": round(stats["wins"] / max(stats["trades"], 1), 4),
            "net_pnl": round(stats["net_pnl"], 2),
        }
    return output


def run_backtest(
    bars: int = 60000,
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    use_mock: bool = True,
    start: str | None = None,
    end: str | None = None,
    decision_start: str | None = None,
    step: int = 1,
) -> dict:
    if os.environ.get("BACKTEST_LIGHT_CONTEXT") == "1":
        import mt5.market as _market_module

        _market_module.momentum_snapshot = lambda *args, **kwargs: {
            "regular_bullish": False,
            "regular_bearish": False,
            "hidden_bullish": False,
            "hidden_bearish": False,
            "latest": None,
            "events": [],
        }
        _market_module.key_level_snapshot = lambda *args, **kwargs: {
            "support": None,
            "resistance": None,
            "zones": [],
        }
        _market_module.cci_divergence_snapshot = lambda *args, **kwargs: {
            "bullish": False,
            "bearish": False,
            "latest": None,
            "events": [],
        }
        _market_module.ict_context_snapshot = lambda *args, **kwargs: {
            "sweep": None,
            "fvg": None,
        }
        _market_module.macd_momentum_snapshot = lambda *args, **kwargs: {
            "state": "neutral",
            "latest_trigger": None,
            "events": [],
            "macd": None,
            "histogram": None,
        }
        _market_module.trend_phase_snapshot = lambda *args, **kwargs: {
            "phase": "neutral",
            "trend": "neutral",
            "reason": "light backtest context disabled",
        }
    db_path = os.environ.get(
        "BACKTEST_DB_PATH", os.path.join(DATA_DIR, "backtest.db")
    )
    try:
        os.remove(db_path)
    except OSError:
        pass
    db = Database(db_path)

    df = _load_history(symbol, timeframe, bars)
    if start:
        df = df[df["time"] >= pd.Timestamp(start)]
    if end:
        df = df[df["time"] <= pd.Timestamp(end)]
    df = df.iloc[:: max(1, int(step))].reset_index(drop=True)
    df = df.reset_index(drop=True)
    decision_floor = pd.Timestamp(decision_start) if decision_start else None
    htf = _build_htf_context(df)
    if len(htf):
        df = pd.merge_asof(
            df.sort_values("time"),
            htf,
            on="time",
            direction="backward",
        )
    df = df.reset_index(drop=True)

    feature_mask = os.environ.get("FEATURE_MASK", "")
    if feature_mask:
        from brain.feature_brain import FeatureBrain

        llm = FeatureBrain(feature_mask)
    elif use_mock:
        llm = MockLLM()
    else:
        llm = None
    trader = TraderAgent(llm)
    advisor = StrategyAdvisor()
    account = _BacktestAccount(INITIAL_BALANCE)
    risk = RiskManager(db, account=account)
    broker = PaperBroker(db)
    equity_curve: list[float] = []
    regime_bars: dict[str, int] = {"trend": 0, "range": 0}
    decision_times: list[str] = []
    warmup = max(BAR_COUNT, 60)

    for i in range(warmup, len(df)):
        bar = df.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        ts = str(bar["time"])
        day = ts[:10]

        events = broker.bar_tick(high, low, close, ts)
        account.on_events(day, ts, events)
        if decision_floor is not None and pd.Timestamp(ts) < decision_floor:
            continue
        decision_times.append(ts)

        window = df.iloc[:i].tail(BAR_COUNT).reset_index(drop=True)
        raw_htf = bar.get("htf_trend")
        htf_context = None
        if raw_htf is not None and not pd.isna(raw_htf):
            htf_context = {
                "trend": str(raw_htf),
                "adx": float(bar.get("htf_adx") or 0),
                "rsi": float(bar.get("htf_rsi") or 50),
            }
        indicators = compute_indicators(window, htf=htf_context)
        market = {
            "df": window,
            "indicators": indicators,
            "symbol": symbol,
            "timeframe": timeframe,
            "close": indicators["close"],
            "atr": indicators["atr"],
        }
        regime_key = indicators.get("market_state", "range")
        regime_bars[regime_key] = regime_bars.get(regime_key, 0) + 1
        strategy = advisor.advise(market, db)
        recent_bars = window.tail(12)
        vision_view = VisionAnalyzer._local_analysis(
            {
                "trend": indicators.get("trend", "neutral"),
                "bars": [
                    {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    }
                    for _, row in recent_bars.iterrows()
                ],
            }
        )
        vision = {"vision": vision_view}
        account.last_price = close
        positions = [
            {
                "side": pos["side"],
                "volume": float(pos["volume"]),
                "entry": float(pos["entry"]),
                "profit": round(
                    (close - float(pos["entry"]))
                    * (1.0 if pos["side"] == "BUY" else -1.0)
                    * float(pos["volume"])
                    * 100,
                    2,
                ),
            }
            for pos in account.open_positions.values()
        ]
        decision = trader.decide(
            market,
            vision=vision,
            strategy=strategy,
            positions=positions,
        )
        review = risk.review(decision, market, positions=positions)
        if review["allow"] and review["order"]:
            order = review["order"]
            order["market_state"] = strategy.get("market_state")
            order["persona"] = strategy.get("preferred_persona")
            if order["action"] == "CLOSE":
                events = broker.close_all(float(bar["close"]), ts=ts)
                account.on_events(day, ts, events)
                continue
            open_price = float(bar["open"])
            spread = symbol_spread(symbol)
            order["entry"] = round(
                open_price + spread
                if order["action"] == "BUY"
                else open_price - spread,
                price_precision(symbol),
            )
            trade = broker.place(order, ts=ts)
            account.on_open(trade)
        equity_curve.append(account.equity(close))

    if equity_curve and len(df):
        last = df.iloc[-1]
        ts = str(last["time"])
        events = broker.close_all(float(last["close"]), ts)
        account.on_events(ts[:10], ts, events)
        equity_curve[-1] = account.equity(float(last["close"]))

    closed = db.closed_trades()
    metrics = _metrics(closed, equity_curve)
    by_regime: dict[str, dict] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "net_pnl": 0.0}
    )
    for trade in closed:
        state = trade.get("market_state") or "未知"
        stats = by_regime[state]
        stats["trades"] += 1
        pnl = float(trade.get("pnl") or 0.0)
        stats["net_pnl"] += pnl
        if pnl > 0:
            stats["wins"] += 1
    regime_trades = {}
    for state, stats in by_regime.items():
        regime_trades[state] = {
            "trades": stats["trades"],
            "wins": stats["wins"],
            "win_rate": round(stats["wins"] / max(stats["trades"], 1), 4),
            "net_pnl": round(stats["net_pnl"], 2),
        }
    report = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": len(decision_times) if decision_times else len(df),
        "bars_per_year": {
            str(year): int(count)
            for year, count in df["time"].dt.year.value_counts().sort_index().items()
        },
        "start": decision_times[0] if decision_times else (
            str(df.iloc[0]["time"]) if len(df) else ""
        ),
        "end": decision_times[-1] if decision_times else (
            str(df.iloc[-1]["time"]) if len(df) else ""
        ),
        "initial_balance": INITIAL_BALANCE,
        "llm": (
            f"feature:{feature_mask}"
            if feature_mask
            else "mock" if use_mock else "real"
        ),
        **metrics,
        "per_year": _year_stats(closed),
        "regimes": {
            "bars": {
                "trend": regime_bars.get("trend", 0),
                "range": regime_bars.get("range", 0),
            },
            "trades": regime_trades,
        },
        "report_time": datetime.now().isoformat(timespec="seconds"),
    }
    stops = [
        abs(float(trade["entry"]) - float(trade["sl"]))
        for trade in closed
        if trade.get("sl")
    ]
    avg_stop = sum(stops) / len(stops) if stops else 0.0
    min_lot_risk_percent = round((avg_stop * 0.01 * 100) / INITIAL_BALANCE * 100, 2)
    report["min_lot_risk_percent"] = min_lot_risk_percent
    report["sparse_years"] = [
        str(year)
        for year, count in df["time"].dt.year.value_counts().items()
        if count < 500
    ]
    if min_lot_risk_percent > MAX_RISK_PERCENT * 100:
        report["risk_note"] = "账户本金过小，最小0.01手使单笔实际风险超过配置上限"

    report_path = os.environ.get(
        "BACKTEST_REPORT_PATH",
        os.path.join(DATA_DIR, "backtest_report.json"),
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2))

    if equity_curve:
        chart_path = os.environ.get(
            "BACKTEST_CHART_PATH",
            os.path.join(DATA_DIR, "backtest_equity.png"),
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(equity_curve, color="#2563eb", linewidth=0.8)
        ax.axhline(INITIAL_BALANCE, color="#94a3b8", linestyle="--", linewidth=0.8)
        ax.set_title(f"AI Trading Agent Backtest - {symbol} {timeframe}")
        ax.set_ylabel("Equity (USD)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=110)
        plt.close(fig)
        report["equity_chart"] = chart_path

    db.conn.close()
    return report
