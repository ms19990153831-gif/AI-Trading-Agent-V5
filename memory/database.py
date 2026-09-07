"""SQLite persistence for decisions, trades, experiences, reviews and heartbeats."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from config import DB_PATH


class Database:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or DB_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, symbol TEXT, timeframe TEXT, action TEXT,
                confidence REAL, reason TEXT, risk_percent REAL,
                sl REAL, tp REAL, volume REAL, state TEXT, cycle INTEGER
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, symbol TEXT, side TEXT, volume REAL,
                entry REAL, sl REAL, tp REAL, sl_current REAL,
                exit_price REAL, exit_ts TEXT, pnl REAL, pnl_pct REAL,
                status TEXT, partial_done INTEGER DEFAULT 0,
                magic INTEGER, reason TEXT, confidence REAL,
                notes TEXT, market_state TEXT, persona TEXT
            );
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, content TEXT, category TEXT
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, summary TEXT, lesson TEXT, trade_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, state TEXT, message TEXT
            );
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, persona TEXT, environments INTEGER,
                trades INTEGER, wins INTEGER, win_rate REAL,
                profit_factor REAL, max_drawdown REAL, net_pnl REAL,
                score REAL, status TEXT
            );
            CREATE TABLE IF NOT EXISTS strategy_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, event_type TEXT, symbol TEXT, timeframe TEXT,
                action TEXT, close REAL, rsi REAL, atr REAL,
                box_low REAL, box_high REAL, two_b_bottom INTEGER,
                two_b_top INTEGER, reason TEXT, outcome TEXT
            );
            """
        )
        # Migration for databases created before the notes column existed.
        try:
            self.conn.execute("ALTER TABLE trades ADD COLUMN notes TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE trades ADD COLUMN ticket INTEGER")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE trades ADD COLUMN market_state TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE trades ADD COLUMN persona TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def save_decision(
        self,
        symbol: str,
        timeframe: str,
        decision: dict,
        state: str = "plan",
        cycle: int = 0,
        volume: float | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO decisions
            (ts, symbol, timeframe, action, confidence, reason, risk_percent,
             sl, tp, volume, state, cycle)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                self._now(),
                symbol,
                timeframe,
                decision.get("action"),
                decision.get("confidence"),
                decision.get("reason"),
                decision.get("risk_percent"),
                decision.get("sl"),
                decision.get("tp"),
                volume,
                state,
                cycle,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_strategy_event(self, event: dict) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO strategy_events
            (ts, event_type, symbol, timeframe, action, close, rsi, atr,
             box_low, box_high, two_b_bottom, two_b_top, reason, outcome)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event.get("ts") or self._now(),
                event.get("event_type"),
                event.get("symbol"),
                event.get("timeframe"),
                event.get("action"),
                event.get("close"),
                event.get("rsi"),
                event.get("atr"),
                event.get("box_low"),
                event.get("box_high"),
                int(bool(event.get("two_b_bottom"))),
                int(bool(event.get("two_b_top"))),
                event.get("reason"),
                event.get("outcome"),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def strategy_events_today(self) -> list[dict]:
        today = datetime.now().date().isoformat()
        rows = self.conn.execute(
            "SELECT * FROM strategy_events WHERE ts LIKE ? ORDER BY id",
            (f"{today}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def strategy_events_since(self, days: int = 7) -> list[dict]:
        since = datetime.now().isoformat(timespec="seconds")
        rows = self.conn.execute(
            "SELECT * FROM strategy_events ORDER BY id DESC LIMIT ?",
            (int(max(1, days) * 200),),
        ).fetchall()
        return [dict(r) for r in rows]

    def decisions_today(self, limit: int = 80) -> list[dict]:
        """Today's AI decision log, oldest first, used by the nightly review."""
        today = datetime.now().date().isoformat()
        rows = self.conn.execute(
            "SELECT ts, action, confidence, reason FROM decisions "
            "WHERE ts LIKE ? ORDER BY id LIMIT ?",
            (f"{today}%", int(max(1, limit))),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_trade(self, trade: dict) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO trades
            (ts, symbol, side, volume, entry, sl, tp, sl_current, status,
             magic, reason, confidence, market_state, persona)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade.get("ts", self._now()),
                trade["symbol"],
                trade["side"],
                trade["volume"],
                trade["entry"],
                trade["sl"],
                trade["tp"],
                trade["sl"],
                "open",
                trade.get("magic", 0),
                trade.get("reason", ""),
                trade.get("confidence", 0),
                trade.get("market_state"),
                trade.get("persona"),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_trade(self, trade_id: int, **fields: Any) -> None:
        allowed = {
            "sl_current",
            "exit_price",
            "exit_ts",
            "pnl",
            "pnl_pct",
            "status",
            "partial_done",
            "volume",
            "notes",
            "ticket",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates)
        self.conn.execute(
            f"UPDATE trades SET {cols} WHERE id=?",
            (*updates.values(), trade_id),
        )
        self.conn.commit()

    def open_trades(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE status='open' AND symbol=?",
                (symbol,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE status='open'"
            ).fetchall()
        return [dict(r) for r in rows]

    def closed_trades(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM trades WHERE status='closed' ORDER BY id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def trades_today(self) -> list[dict]:
        """Trades opened or closed today (local date), oldest first."""
        today = datetime.now().date().isoformat()
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE ts LIKE ? OR exit_ts LIKE ? ORDER BY ts",
            (f"{today}%", f"{today}%"),
        ).fetchall()
        return [dict(r) for r in rows]

    def trades_current_session(self) -> list[dict]:
        """Trades opened in the current 06:00-05:00 trading session."""
        now = datetime.now()
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now.hour < 6:
            start -= timedelta(days=1)
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE ts >= ? ORDER BY ts",
            (start.isoformat(timespec="seconds"),),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_experience(self, content: str, category: str = "lesson") -> int:
        cur = self.conn.execute(
            "INSERT INTO experiences (ts, content, category) VALUES (?,?,?)",
            (self._now(), content, category),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_recent_experiences(self, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiences ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_review(self, summary: str, lesson: str, trade_count: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO reviews (ts, summary, lesson, trade_count) VALUES (?,?,?,?)",
            (self._now(), summary, lesson, trade_count),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_heartbeat(self, state: str, message: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO heartbeats (ts, state, message) VALUES (?,?,?)",
            (self._now(), state, message),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_training_run(
        self,
        persona: str,
        environments: int,
        metrics: dict,
        score: float,
        status: str = "ok",
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO training_runs
            (ts, persona, environments, trades, wins, win_rate,
             profit_factor, max_drawdown, net_pnl, score, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                self._now(),
                persona,
                int(environments),
                int(metrics.get("trades", 0)),
                int(metrics.get("wins", 0)),
                float(metrics.get("win_rate", 0)),
                float(metrics.get("profit_factor", 0)),
                float(metrics.get("max_drawdown", 0)),
                float(metrics.get("net_pnl", 0)),
                float(score),
                status,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def daily_pnl(self) -> float:
        today = datetime.now().date().isoformat()
        rows = self.conn.execute(
            "SELECT pnl FROM trades WHERE status='closed' AND exit_ts LIKE ?",
            (f"{today}%",),
        ).fetchall()
        return float(sum((r["pnl"] or 0.0) for r in rows))

    def daily_pnl_session(self) -> float:
        """Realized PnL of the 06:00-05:00 trading session.

        The calendar-day version resets at local midnight, which lets the
        daily loss limit reopen mid-session. This method resets only at the
        session boundary defined by MARKET_OPEN_TIME/MARKET_CLOSE_TIME.
        """
        now = datetime.now()
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now.hour < 6:
            start -= timedelta(days=1)
        end = start + timedelta(hours=23)
        rows = self.conn.execute(
            "SELECT pnl, exit_ts FROM trades WHERE status='closed'"
        ).fetchall()
        total = 0.0
        for row in rows:
            value = row["exit_ts"]
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(
                    str(value).replace("Z", "+00:00")
                )
            except ValueError:
                continue
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(
                    timezone(timedelta(hours=8))
                ).replace(tzinfo=None)
            if start <= parsed < end:
                total += float(row["pnl"] or 0.0)
        return total

    def reviews_today(self) -> int:
        today = datetime.now().date().isoformat()
        row = self.conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE ts LIKE ?",
            (f"{today}%",),
        ).fetchone()
        return int(row[0]) if row else 0

    def loss_streak(self) -> int:
        rows = self.conn.execute(
            "SELECT pnl FROM trades WHERE status='closed' ORDER BY id DESC LIMIT 20"
        ).fetchall()
        streak = 0
        for row in rows:
            if (row["pnl"] or 0.0) < 0:
                streak += 1
            else:
                break
        return streak
