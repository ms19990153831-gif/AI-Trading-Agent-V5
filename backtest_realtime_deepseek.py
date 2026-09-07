"""Real-granularity DeepSeek backtest on M1 bars.

AI decisions happen only on completed H1 bars, but every open position is
managed on real M1 bars so SL/TP/partial fills use actual minute prices.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.strategy_advisor import StrategyAdvisor  # noqa: E402
from agent.trader_agent import TraderAgent  # noqa: E402
from brain.llm import MockLLM  # noqa: E402
from backtest import _BacktestAccount, _metrics  # noqa: E402
from execution.paper_broker import PaperBroker  # noqa: E402
from memory.database import Database  # noqa: E402
from mt5.market import compute_indicators  # noqa: E402
from risk.manager import RiskManager  # noqa: E402
from vision.analyzer import VisionAnalyzer  # noqa: E402
from config import INITIAL_BALANCE, MAX_RISK_PERCENT


def _load(path: str, kind: str = "h1") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def run_m1_deepseek_backtest(
    m1_path: str,
    h1_path: str,
    decision_start: str = "2026-08-28",
    end: str = "2026-09-03 12:50:00",
) -> dict:
    m1 = _load(m1_path, "m1")
    h1 = _load(h1_path, "h1")
    h1 = h1[h1["time"] <= pd.Timestamp(end)].reset_index(drop=True)
    m1 = m1[m1["time"] <= pd.Timestamp(end)].reset_index(drop=True)
    m1["hour"] = m1["time"].dt.floor("1h")
    h1_start = pd.Timestamp(decision_start)
    db_path = os.path.join(ROOT, "data", "backtest_realtime.db")
    try:
        os.remove(db_path)
    except OSError:
        pass
    db = Database(db_path)
    broker = PaperBroker(db)
    account = _BacktestAccount(INITIAL_BALANCE)
    risk = RiskManager(db, account=account)
    trader = (
        TraderAgent(MockLLM())
        if os.environ.get("USE_MOCK_BACKTEST") == "1"
        else TraderAgent()
    )
    advisor = StrategyAdvisor()

    pending_order = None
    equity: list[float] = []
    closed: list[dict] = []
    h1_used = 0

    # Walk every real M1 bar and decide on:
    # 1) completed H1 close, 2) position state change, 3) 0.8 x ATR range wake.
    def position_state_key() -> tuple:
        return tuple(
            sorted(str(pos_id) for pos_id in account.open_positions)
        )

    def decide_now(
        hour_for_h1: pd.Timestamp,
        current_price: float,
        reason: str,
        now: pd.Timestamp | None = None,
    ) -> None:
        nonlocal pending_order, h1_used, cooldown_until
        if (
            reason != "h1_close"
            and cooldown_until is not None
            and now is not None
            and now < cooldown_until
        ):
            return
        if hour_for_h1 < h1_start:
            return
        upto = h1[h1["time"] <= hour_for_h1].tail(200).reset_index(drop=True)
        if len(upto) < 60:
            return
        indicators = compute_indicators(upto)
        atr = float(indicators.get("atr") or max(current_price * 0.002, 1.0))
        market = {
            "df": upto,
            "indicators": indicators,
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "close": indicators["close"],
            "atr": indicators["atr"],
        }
        if risk.hard_pause_reason(account.snapshot()):
            return
        recent = upto.tail(12)
        vision = {
            "vision": VisionAnalyzer._local_analysis(
                {
                    "trend": indicators.get("trend", "neutral"),
                    "bars": [
                        {
                            "open": float(r["open"]),
                            "high": float(r["high"]),
                            "low": float(r["low"]),
                            "close": float(r["close"]),
                        }
                        for _, r in recent.iterrows()
                    ],
                }
            )
        }
        strategy = advisor.advise(market, db)
        positions = [
            {
                "side": pos["side"],
                "volume": float(pos["volume"]),
                "entry": float(pos["entry"]),
                "profit": round(
                    (current_price - float(pos["entry"]))
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
            macro={},
            strategy=strategy,
            positions=positions,
        )
        review = risk.review(decision, market, positions=positions)
        if review["allow"] and review["order"]:
            order = review["order"]
            order["market_state"] = strategy.get("market_state")
            order["persona"] = strategy.get("preferred_persona")
            order["spread"] = 0.0
            pending_order = order
            print(
                f"{hour_for_h1} {reason} {order['action']} "
                f"entry_ref={order.get('entry')} conf={order.get('confidence')} "
                f"reason={str(order.get('reason'))[:80]}"
            )
        h1_used += 1
        cooldown_until = (now or pd.Timestamp(hour_for_h1)) + pd.Timedelta(
            minutes=5
        )

    rows = m1.to_dict("records")
    cooldown_until = None
    last_hour = None
    last_state = position_state_key()
    wake_high = None
    wake_low = None
    wake_atr = 1.0
    decided_this_bar = False

    for index, row in enumerate(rows):
        current_time = row["time"]
        hour = row["hour"]
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        ts = str(current_time)
        day = current_time.strftime("%Y-%m-%d")
        decided_this_bar = False

        # A new hour means the previous hour is now a completed H1.
        hour_closed = last_hour is not None and hour != last_hour

        # Fill any pending order at this M1 open before touching positions.
        if pending_order is not None:
            order = pending_order
            entry = open_price
            if order["action"] == "BUY":
                entry = round(open_price + float(order.get("spread", 0.0)), 2)
            elif order["action"] == "SELL":
                entry = round(open_price - float(order.get("spread", 0.0)), 2)
            if order["action"] == "CLOSE":
                events = broker.close_all(open_price, ts=ts)
                account.on_events(day, ts, events)
            else:
                order["entry"] = entry
                trade = broker.place(order, ts=ts)
                account.on_open(trade)
            pending_order = None
            decided_this_bar = True

        # Real M1 position management.
        before_state = position_state_key()
        events = broker.bar_tick(high, low, close, ts)
        account.on_events(day, ts, events)
        after_state = position_state_key()
        account.last_price = close
        equity.append(account.equity(close))

        # Range wake tracking since the last AI decision.
        if wake_high is None:
            wake_high = high
            wake_low = low
        else:
            wake_high = max(wake_high, high)
            wake_low = min(wake_low, low)

        if not decided_this_bar:
            if hour_closed:
                # Completed H1 whose open time is last_hour.
                decide_now(last_hour, close, "h1_close", now=current_time)
                decided_this_bar = True
            elif events:
                # Position changed or partial TP happened, wake immediately.
                completed = hour - pd.Timedelta(hours=1)
                decide_now(completed, close, "position_change", now=current_time)
                decided_this_bar = True
            elif (
                wake_high is not None
                and wake_low is not None
                and (wake_high - wake_low) >= 0.8 * wake_atr
            ):
                completed = hour - pd.Timedelta(hours=1)
                decide_now(completed, close, "range_wake", now=current_time)
                decided_this_bar = True

        if decided_this_bar:
            # Reset wake baseline after every AI decision.
            upto = h1[h1["time"] <= (hour - pd.Timedelta(hours=1))].tail(200)
            if len(upto) >= 60:
                wake_atr = float(
                    compute_indicators(upto).get("atr")
                    or max(close * 0.002, 1.0)
                )
            wake_high = high
            wake_low = low
            last_state = after_state

        last_hour = hour

    if rows:
        last_row = rows[-1]
        ts = str(last_row["time"])
        events = broker.close_all(float(last_row["close"]), ts)
        account.on_events(ts[:10], ts, events)
        if equity:
            equity[-1] = account.equity(float(last_row["close"]))

    closed = db.closed_trades()
    metrics = _metrics(closed, equity)
    metrics["h1_decisions"] = h1_used
    metrics["initial_balance"] = INITIAL_BALANCE
    return metrics


if __name__ == "__main__":
    result = run_m1_deepseek_backtest(
        os.path.join(ROOT, "data", "history_XAUUSD_M1_fresh_5d.csv"),
        os.path.join(ROOT, "data", "history_XAUUSD_H1_fresh.csv"),
        decision_start="2026-08-28",
        end="2026-09-03 12:50:00",
    )
    print("RESULT", result)
