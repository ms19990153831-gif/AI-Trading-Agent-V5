"""Market hours gate tests."""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from market_hours import market_is_open, next_open, preopen_time  # noqa: E402


class MarketHoursTest(unittest.TestCase):
    def test_monday_before_open_closed(self) -> None:
        self.assertFalse(market_is_open(datetime(2026, 8, 24, 3, 0)))

    def test_monday_open(self) -> None:
        self.assertTrue(market_is_open(datetime(2026, 8, 24, 6, 0)))

    def test_daily_break_closed(self) -> None:
        self.assertFalse(market_is_open(datetime(2026, 8, 25, 5, 30)))

    def test_weekend_closed(self) -> None:
        self.assertFalse(market_is_open(datetime(2026, 8, 22, 12, 0)))

    def test_next_open_after_break(self) -> None:
        self.assertEqual(
            next_open(datetime(2026, 8, 25, 5, 30)),
            datetime(2026, 8, 25, 6, 0),
        )

    def test_preopen_time(self) -> None:
        self.assertEqual(
            preopen_time(datetime(2026, 8, 25, 5, 30)),
            datetime(2026, 8, 25, 5, 55),
        )


if __name__ == "__main__":
    unittest.main()
