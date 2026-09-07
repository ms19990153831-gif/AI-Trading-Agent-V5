"""Account snapshot built from closed paper trades (or MT5 when configured)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import EXECUTION_MODE, INITIAL_BALANCE, MAX_LOSS_STREAK, MT5_MAGIC


class Account:
    def __init__(self, db, force_paper: bool = False) -> None:
        self.db = db
        self.force_paper = force_paper

    def snapshot(self) -> dict:
        streak, paused = self._loss_info()
        if EXECUTION_MODE == "mt5" and not self.force_paper:
            import MetaTrader5 as mt5

            account = mt5.account_info()
            if account is not None:
                positions = mt5.positions_get() or []
                my_positions = [p for p in positions if p.magic == MT5_MAGIC]
                return {
                    "balance": round(account.balance, 2),
                    "equity": round(account.equity, 2),
                    "open_positions": len(my_positions),
                    "daily_pnl": round(self.db.daily_pnl_session(), 2),
                    "loss_streak": streak,
                    "loss_paused": paused,
                    "positions": [
                        {
                            "side": (
                                "BUY"
                                if p.type == 0
                                else "SELL"
                            ),
                            "volume": float(p.volume),
                            "entry": float(p.price_open),
                            "profit": float(p.profit or 0.0),
                            "symbol": p.symbol,
                            "ticket": int(p.ticket),
                        }
                        for p in my_positions
                    ],
                }
        closed = self.db.closed_trades()
        pnl = sum(float(t["pnl"] or 0.0) for t in closed)
        balance = INITIAL_BALANCE + pnl
        open_positions = self.db.open_trades()
        return {
            "balance": round(balance, 2),
            "equity": round(balance, 2),
            "open_positions": len(open_positions),
            "daily_pnl": round(self.db.daily_pnl_session(), 2),
            "loss_streak": streak,
            "loss_paused": paused,
            "positions": [
                {
                    "side": trade["side"],
                    "volume": float(trade["volume"]),
                    "entry": float(trade["entry"]),
                    "profit": 0.0,
                    "symbol": trade.get("symbol"),
                    "ticket": trade.get("ticket"),
                }
                for trade in open_positions
            ],
        }

    def _loss_info(self) -> tuple[int, bool]:
        rows = self.db.closed_trades(limit=50)
        streak = 0
        last_loss = None
        for trade in rows:
            pnl = float(trade.get("pnl") or 0.0)
            if pnl < 0:
                streak += 1
                last_loss = trade.get("exit_ts") or last_loss
            else:
                break
        paused = False
        if last_loss and streak >= MAX_LOSS_STREAK:
            try:
                last = self._parse_ts(last_loss)
                now = (
                    datetime.now(timezone.utc)
                    if last.tzinfo is not None
                    else datetime.now()
                )
                paused = (now - last).total_seconds() < 24 * 3600
            except ValueError:
                paused = True
        if not paused and streak >= MAX_LOSS_STREAK:
            streak = 0
        return streak, paused

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        """Parse an ISO timestamp, keeping naive timestamps naive."""
        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text)
