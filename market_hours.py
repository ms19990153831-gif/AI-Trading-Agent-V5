"""Local market-hours gate for live trading modes.

Assumes a 24/5 metal/forex schedule with a one-hour daily break: the market
opens at MARKET_OPEN_TIME and closes at MARKET_CLOSE_TIME the next morning.
Saturday and Sunday are closed.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from config import (
    MARKET_CLOSE_TIME,
    MARKET_HOURS_ENABLED,
    MARKET_OPEN_TIME,
    PRE_OPEN_MINUTES,
)


def _parse_hhmm(text: str) -> tuple[int, int]:
    hour_text, minute_text = str(text).strip().split(":")
    return int(hour_text), int(minute_text)


OPEN_HOUR, OPEN_MINUTE = _parse_hhmm(MARKET_OPEN_TIME)
CLOSE_HOUR, CLOSE_MINUTE = _parse_hhmm(MARKET_CLOSE_TIME)
PRE_OPEN = timedelta(minutes=max(0, int(PRE_OPEN_MINUTES)))


def open_at(day: datetime) -> datetime:
    return day.replace(hour=OPEN_HOUR, minute=OPEN_MINUTE, second=0, microsecond=0)


def close_after(day: datetime) -> datetime:
    return (day + timedelta(days=1)).replace(
        hour=CLOSE_HOUR, minute=CLOSE_MINUTE, second=0, microsecond=0
    )


def market_is_open(now: datetime | None = None) -> bool:
    if not MARKET_HOURS_ENABLED:
        return True
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    # A session that started the previous weekday is still running before
    # today's open (e.g. Tuesday 03:00 belongs to Monday's 06:00-05:00 session).
    for day in (now - timedelta(days=1), now):
        if day.weekday() >= 5:
            continue
        if open_at(day) <= now < close_after(day):
            return True
    return False


def next_open(now: datetime | None = None) -> datetime:
    now = now or datetime.now()
    for offset in range(0, 8):
        day = (now + timedelta(days=offset)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        candidate = open_at(day)
        if candidate > now and market_is_open(candidate):
            return candidate
    return open_at(now + timedelta(days=8))


def preopen_time(now: datetime | None = None) -> datetime:
    return next_open(now) - PRE_OPEN
