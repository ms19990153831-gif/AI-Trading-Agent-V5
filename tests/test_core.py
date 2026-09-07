"""Core unit tests: mock brain, firewall, order sizing, paper broker and
indicator math."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.llm import MockLLM  # noqa: E402
from brain.feature_brain import FeatureBrain  # noqa: E402
from agent.ceo import TradingCEO  # noqa: E402
from agent.strategy_advisor import StrategyAdvisor  # noqa: E402
from agent.trader_agent import TraderAgent  # noqa: E402
from execution.order_manager import OrderManager  # noqa: E402
from execution.paper_broker import PaperBroker  # noqa: E402
from memory.database import Database  # noqa: E402
from mt5.market import compute_indicators  # noqa: E402
from risk.manager import RiskManager  # noqa: E402
from risk.firewall import Firewall  # noqa: E402
from state_machine import State, StateMachine  # noqa: E402
from tools.calculator import normalize_risk  # noqa: E402
from tools.market_context import (  # noqa: E402
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
from tools.market_structure import (  # noqa: E402
    structure_reference_text,
    structure_snapshot,
)
from tools.quality import entry_quality  # noqa: E402
from tools.structure import (  # noqa: E402
    boundary_guard,
    close_signal,
    extreme_chase_guard,
    reversal_signal,
    swing_reversal_signal,
    trend_conflict_guard,
    vision_reversal_guard,
)


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        handle.close()
        self.path = handle.name
        self.db = Database(self.path)

    def tearDown(self) -> None:
        self.db.conn.close()
        try:
            os.remove(self.path)
        except OSError:
            pass

    def test_strategy_events_roundtrip(self) -> None:
        self.db.save_strategy_event(
            {
                "event_type": "boundary_guard",
                "symbol": "XAUUSD",
                "timeframe": "M15",
                "action": "SELL",
                "close": 4576.44,
                "rsi": 24.3,
                "atr": 18.0,
                "box_low": 4575.0,
                "box_high": 4620.0,
                "two_b_bottom": True,
                "two_b_top": False,
                "reason": "箱体边界保护",
                "outcome": "blocked",
            }
        )
        rows = self.db.strategy_events_today()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_type"], "boundary_guard")
        self.assertEqual(rows[0]["outcome"], "blocked")

    def test_strategy_events_summary(self) -> None:
        from agent.reflection_agent import summarize_strategy_events

        text = summarize_strategy_events(
            [
                {"event_type": "boundary_guard", "action": "SELL", "outcome": "blocked"},
                {"event_type": "reversal_signal", "action": "BUY", "outcome": "blocked"},
            ]
        )
        self.assertIn("边界保护拦截1次", text)
        self.assertIn("2B反转信号1次", text)

    def test_decisions_today_roundtrip(self) -> None:
        self.db.save_decision(
            "XAUUSD",
            "H1",
            {
                "action": "WAIT",
                "confidence": 55,
                "reason": "结构、关键位与CCI共同指向等待",
                "risk_percent": 0.01,
            },
            state="planned",
            cycle=1,
        )
        rows = self.db.decisions_today()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "WAIT")
        self.assertIn("CCI", rows[0]["reason"])

    def test_trades_current_session_filters_old_trades(self) -> None:
        base = {
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 0.01,
            "entry": 100.0,
            "sl": 99.0,
            "tp": 103.0,
            "magic": 1,
        }
        now = datetime.now().isoformat(timespec="seconds")
        old = (datetime.now() - timedelta(days=3)).isoformat(timespec="seconds")
        self.db.save_trade({**base, "ts": now})
        self.db.save_trade({**base, "ts": old})
        rows = self.db.trades_current_session()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ts"], now)

    def test_daily_pnl_session_resets_at_session_boundary(self) -> None:
        now = datetime.now()
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now.hour < 6:
            start -= timedelta(days=1)
        base = {
            "symbol": "XAUUSD",
            "side": "BUY",
            "volume": 0.01,
            "entry": 4300.0,
            "sl": 4290.0,
            "tp": 4350.0,
            "magic": 1,
        }
        in_session_id = self.db.save_trade(
            {**base, "ts": (start + timedelta(hours=1)).isoformat()}
        )
        old_session_id = self.db.save_trade(
            {**base, "ts": (start - timedelta(hours=1)).isoformat()}
        )
        self.db.update_trade(
            in_session_id,
            status="closed",
            exit_ts=(start + timedelta(hours=3)).isoformat(),
            pnl=-40.0,
        )
        self.db.update_trade(
            old_session_id,
            status="closed",
            exit_ts=(start - timedelta(hours=3)).isoformat(),
            pnl=-100.0,
        )
        self.assertEqual(self.db.daily_pnl_session(), -40.0)


class MockBrainTest(unittest.TestCase):
    def test_bullish_decision(self) -> None:
        llm = MockLLM()
        decision = llm.ask_json(
            "trader",
            'CONTEXT_JSON: {"trend":"bullish","rsi":45,"close":3350,"atr":4}',
        )
        self.assertEqual(decision["action"], "BUY")
        self.assertGreaterEqual(decision["confidence"], 70)

    def test_bearish_decision(self) -> None:
        llm = MockLLM()
        decision = llm.ask_json(
            "trader",
            'CONTEXT_JSON: {"trend":"bearish","rsi":55,"close":3350,"atr":4}',
        )
        self.assertEqual(decision["action"], "SELL")

    def test_range_mean_reversion_decision(self) -> None:
        llm = MockLLM()
        decision = llm.ask_json(
            "trader",
            'CONTEXT_JSON: {"trend":"bullish","rsi":75,"close":3350,"atr":4,'
            '"strategy_advisor":{"market_state":"震荡"}}',
        )
        self.assertEqual(decision["action"], "SELL")
        self.assertGreaterEqual(decision["confidence"], 60)

    def test_trend_momentum_forced_decision(self) -> None:
        llm = MockLLM()
        decision = llm.ask_json(
            "trader",
            'CONTEXT_JSON: {"trend":"bullish","rsi":60,"close":3350,"atr":4,'
            '"market_state":"trend","momentum_ok":true}',
        )
        self.assertEqual(decision["action"], "BUY")
        self.assertGreaterEqual(decision["confidence"], 60)

    def test_range_signal_forced_decision(self) -> None:
        llm = MockLLM()
        decision = llm.ask_json(
            "trader",
            'CONTEXT_JSON: {"trend":"bullish","rsi":72,"close":3350,"atr":4,'
            '"market_state":"range","range_signal":"SELL"}',
        )
        self.assertEqual(decision["action"], "SELL")
        self.assertGreaterEqual(decision["confidence"], 60)

    def test_chinese_review_routing(self) -> None:
        llm = MockLLM()
        result = llm.ask_json(
            "复盘AI",
            'CONTEXT_JSON: {"total":2,"wins":1,"pnl":10}\n'
            "Review today's trading and write one lesson.",
        )
        self.assertIn("lesson", result)

    def test_trader_routing_with_chinese_prompt(self) -> None:
        from brain.prompt import TRADER_SYSTEM

        llm = MockLLM()
        result = llm.ask_json(
            TRADER_SYSTEM,
            'CONTEXT_JSON: {"trend":"bullish","rsi":60,"close":4600,"atr":19}'
            "\nReturn your trading decision as strict JSON.",
        )
        self.assertIn("action", result)

    def test_close_when_multi_confirmation_present(self) -> None:
        llm = MockLLM()
        decision = llm.ask_json(
            "trader",
            'CONTEXT_JSON: {"positions":[{"side":"SELL","entry":4590}],'
            '"close_signal":{"action":"CLOSE","confidence":80,'
            '"reason":"持仓反向确认"}}',
        )
        self.assertEqual(decision["action"], "CLOSE")


class FirewallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.firewall = Firewall()
        self.account = {
            "balance": 10000,
            "open_positions": 0,
            "daily_pnl": 0.0,
            "loss_streak": 0,
        }
        self._patches = [
            mock.patch("tools.structure.ENABLE_BOUNDARY_GUARD", True),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()

    def test_allows_good_trade(self) -> None:
        result = self.firewall.check(
            {"action": "BUY", "confidence": 80, "risk_percent": 0.01},
            self.account,
        )
        self.assertTrue(result["allow"])

    def test_blocks_low_confidence(self) -> None:
        result = self.firewall.check(
            {"action": "BUY", "confidence": 55, "risk_percent": 0.01},
            self.account,
        )
        self.assertFalse(result["allow"])

    def test_trend_state_allows_lower_confidence(self) -> None:
        result = self.firewall.check(
            {
                "action": "BUY",
                "confidence": 62,
                "risk_percent": 0.01,
                "market_state": "趋势",
            },
            self.account,
        )
        self.assertTrue(result["allow"])

    def test_range_state_allows_lower_confidence(self) -> None:
        result = self.firewall.check(
            {
                "action": "BUY",
                "confidence": 62,
                "risk_percent": 0.01,
                "market_state": "震荡",
            },
            self.account,
        )
        self.assertTrue(result["allow"])

    def test_blocks_daily_loss_limit(self) -> None:
        account = {**self.account, "daily_pnl": -400.0}
        with mock.patch("risk.firewall.ENABLE_DAILY_LOSS_LIMIT", True), mock.patch(
            "risk.firewall.DAILY_LOSS_LIMIT", 0.03
        ):
            result = self.firewall.check(
                {"action": "BUY", "confidence": 85, "risk_percent": 0.01},
                account,
            )
        self.assertFalse(result["allow"])

    def test_blocks_loss_pause(self) -> None:
        account = {**self.account, "loss_paused": True}
        with mock.patch("risk.firewall.ENABLE_LOSS_PAUSE", True):
            result = self.firewall.check(
                {"action": "BUY", "confidence": 90, "risk_percent": 0.01},
                account,
            )
        self.assertFalse(result["allow"])

    def test_daily_loss_limit_can_be_disabled(self) -> None:
        account = {**self.account, "daily_pnl": -400.0}
        with mock.patch("risk.firewall.ENABLE_DAILY_LOSS_LIMIT", False):
            result = self.firewall.check(
                {"action": "BUY", "confidence": 85, "risk_percent": 0.01},
                account,
            )
        self.assertTrue(result["allow"])

    def test_loss_pause_can_be_disabled(self) -> None:
        account = {**self.account, "loss_paused": True}
        with mock.patch("risk.firewall.ENABLE_LOSS_PAUSE", False):
            result = self.firewall.check(
                {"action": "BUY", "confidence": 90, "risk_percent": 0.01},
                account,
            )
        self.assertTrue(result["allow"])

    def test_blocks_sell_at_box_bottom_oversold(self) -> None:
        market = {
            "indicators": {
                "close": 4576.44,
                "atr": 18.0,
                "rsi": 24.3,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
            }
        }
        result = self.firewall.check(
            {"action": "SELL", "confidence": 80, "risk_percent": 0.01},
            self.account,
            market,
        )
        self.assertFalse(result["allow"])
        self.assertIn("箱体边界保护", "; ".join(result["reasons"]))

    def test_allows_buy_at_box_bottom_oversold(self) -> None:
        market = {
            "indicators": {
                "close": 4576.44,
                "atr": 18.0,
                "rsi": 24.3,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
            }
        }
        result = self.firewall.check(
            {"action": "BUY", "confidence": 80, "risk_percent": 0.01},
            self.account,
            market,
        )
        self.assertTrue(result["allow"])

    def test_blocks_buy_at_box_top_overbought(self) -> None:
        market = {
            "indicators": {
                "close": 4618.0,
                "atr": 18.0,
                "rsi": 71.5,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
            }
        }
        result = self.firewall.check(
            {"action": "BUY", "confidence": 80, "risk_percent": 0.01},
            self.account,
            market,
        )
        self.assertFalse(result["allow"])
        self.assertIn("箱体边界保护", "; ".join(result["reasons"]))

    def test_allows_close_with_multi_confirmation(self) -> None:
        account = {
            **self.account,
            "open_positions": 1,
            "positions": [{"side": "SELL", "volume": 0.01, "entry": 4590.0}],
        }
        market = {
            "indicators": {
                "close": 4576.44,
                "atr": 18.0,
                "rsi": 40.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": True,
                "two_b_top": False,
            }
        }
        result = self.firewall.check(
            {"action": "CLOSE", "confidence": 80, "risk_percent": 0.01},
            account,
            market,
        )
        self.assertTrue(result["allow"])

    def test_blocks_close_without_positions(self) -> None:
        result = self.firewall.check(
            {"action": "CLOSE", "confidence": 80, "risk_percent": 0.01},
            self.account,
        )
        self.assertFalse(result["allow"])

    def test_allows_close_with_positions_and_confidence(self) -> None:
        account = {
            **self.account,
            "open_positions": 1,
            "positions": [{"side": "SELL", "volume": 0.01, "entry": 4590.0}],
        }
        market = {
            "indicators": {
                "close": 4576.44,
                "atr": 18.0,
                "rsi": 40.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": False,
                "two_b_top": False,
            }
        }
        result = self.firewall.check(
            {"action": "CLOSE", "confidence": 80, "risk_percent": 0.01},
            account,
            market,
        )
        self.assertTrue(result["allow"])

    def test_blocks_close_low_confidence(self) -> None:
        account = {
            **self.account,
            "open_positions": 1,
            "positions": [{"side": "SELL", "volume": 0.01, "entry": 4590.0}],
        }
        market = {
            "indicators": {
                "close": 4576.44,
                "atr": 18.0,
                "rsi": 40.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": True,
                "two_b_top": False,
            }
        }
        result = self.firewall.check(
            {"action": "CLOSE", "confidence": 50, "risk_percent": 0.01},
            account,
            market,
        )
        self.assertFalse(result["allow"])


class TraderBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            mock.patch("tools.structure.ENABLE_BOUNDARY_GUARD", True),
            mock.patch("tools.structure.ENABLE_AI_CLOSE", True),
            mock.patch("agent.trader_agent.ENABLE_REVERSAL_SIGNAL", True),
            mock.patch("tools.quality.ENABLE_QUALITY_GATE", True),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()

    @staticmethod
    def _market(**overrides: object) -> dict:
        indicators = {
            "close": 4576.44,
            "rsi": 24.3,
            "atr": 18.0,
            "trend": "bearish",
            "market_state": "trend",
            "momentum_ok": True,
            "range_signal": "none",
            "htf_trend": "bullish",
            "htf_adx": 25.0,
            "htf_rsi": 55.0,
            "donchian_high": 4620.0,
            "donchian_low": 4575.0,
            "box_zone_top": 4609.2,
            "box_zone_bottom": 4585.8,
            "two_b_bottom": False,
            "two_b_top": False,
        }
        indicators.update(overrides)
        return {"symbol": "XAUUSD", "timeframe": "M15", "indicators": indicators}

    def test_trend_sell_at_box_bottom_keeps_guard(self) -> None:
        trader = TraderAgent(MockLLM())
        decision = trader.decide(self._market())
        # 边界保护 + 质量门槛都会拦下箱底超卖追空，最终转为 WAIT。
        self.assertEqual(decision["action"], "WAIT")
        self.assertIsNotNone(decision.get("boundary_guard"))
        self.assertIsNotNone(decision.get("quality_gate"))

    def test_two_b_bottom_overrides_to_buy(self) -> None:
        trader = TraderAgent(MockLLM())
        decision = trader.decide(self._market(two_b_bottom=True))
        self.assertEqual(decision["action"], "BUY")
        self.assertIsNone(decision.get("boundary_guard"))

    def test_opposite_direction_closes_held_position(self) -> None:
        class _BuyLLM:
            def ask_json(self, system, user):
                return {
                    "action": "BUY",
                    "confidence": 72,
                    "risk_percent": 0.01,
                    "reason": "趋势看涨",
                }

        trader = TraderAgent(_BuyLLM())
        market = self._market(
            trend="bullish",
            market_state="trend",
            momentum_ok=True,
            two_b_bottom=False,
        )
        decision = trader.decide(
            market,
            positions=[{"side": "SELL", "volume": 0.01, "entry": 4612.52}],
        )
        self.assertEqual(decision["action"], "CLOSE")

    def test_close_hint_converts_wait_to_close(self) -> None:
        trader = TraderAgent(MockLLM())
        market = self._market(
            trend="bullish",
            market_state="trend",
            momentum_ok=True,
            rsi=64.5,
            sma20=4598.63,
            close=4604.78,
            two_b_bottom=False,
        )
        decision = trader.decide(
            market,
            positions=[{"side": "SELL", "volume": 0.01, "entry": 4612.52}],
        )
        self.assertEqual(decision["action"], "CLOSE")

    def test_risk_percent_is_clamped_to_max(self) -> None:
        class _HighRiskLLM:
            def ask_json(self, system, user):
                return {
                    "action": "BUY",
                    "confidence": 80,
                    "risk_percent": 0.5,
                    "reason": "测试重仓",
                }

        with mock.patch("agent.trader_agent.MAX_RISK_PERCENT", 0.01):
            trader = TraderAgent(_HighRiskLLM())
            decision = trader.decide(self._market())
        self.assertEqual(decision["risk_percent"], 0.01)

    @mock.patch("tools.structure.ENABLE_VISION_GUARD", True)
    def test_vision_stop_blocks_new_sell_when_no_position(self) -> None:
        class _SellLLM:
            def ask_json(self, system, user):
                return {
                    "action": "SELL",
                    "confidence": 75,
                    "risk_percent": 0.01,
                    "reason": "下跌趋势",
                }

        trader = TraderAgent(_SellLLM())
        market = self._market(
            close=4600.0,
            rsi=50.0,
            donchian_high=4640.0,
            donchian_low=4570.0,
            box_zone_top=4629.2,
            box_zone_bottom=4580.8,
        )
        decision = trader.decide(
            market,
            vision={
                "vision": {
                    "structure": "H1止跌K线（长下影/阳包阴）",
                    "setup": "停止追空，观察企稳",
                }
            },
        )
        self.assertEqual(decision["action"], "WAIT")
        self.assertIsNotNone(decision.get("vision_guard"))
        self.assertTrue(
            any(
                event.get("event_type") == "vision_guard"
                for event in decision.get("strategy_events", [])
            )
        )

    @mock.patch("tools.structure.ENABLE_VISION_GUARD", True)
    def test_vision_stop_closes_existing_sell(self) -> None:
        class _SellLLM:
            def ask_json(self, system, user):
                return {
                    "action": "SELL",
                    "confidence": 75,
                    "risk_percent": 0.01,
                    "reason": "下跌趋势",
                }

        trader = TraderAgent(_SellLLM())
        market = self._market(
            close=4600.0,
            rsi=50.0,
            donchian_high=4640.0,
            donchian_low=4570.0,
            box_zone_top=4629.2,
            box_zone_bottom=4580.8,
        )
        decision = trader.decide(
            market,
            vision={
                "vision": {
                    "structure": "H1止跌K线（长下影/阳包阴）",
                    "setup": "停止追空，观察企稳",
                }
            },
            positions=[{"side": "SELL", "volume": 0.01, "entry": 4612.52}],
        )
        self.assertEqual(decision["action"], "CLOSE")
        self.assertIn("止跌", decision["reason"])

    def test_trend_conflict_blocks_sell_against_bullish_major(self) -> None:
        indicators = {
            "close": 4434.0,
            "atr": 14.0,
            "rsi": 76.0,
            "adx": 30.0,
            "market_state": "trend",
            "trend": "bullish",
            "donchian_high": 4460.0,
            "donchian_low": 4300.0,
            "structure": {
                "trend": "bearish",
                "choch_up": False,
                "choch_down": False,
                "health": 48,
                "health_state": "HOLDING",
            },
        }
        with mock.patch("tools.structure.ENABLE_TREND_CONFLICT_GUARD", True):
            guard = trend_conflict_guard("SELL", indicators)
        self.assertIsNotNone(guard)
        self.assertTrue(guard["block"])
        self.assertIn("趋势冲突", guard["reason"])

    def test_trend_conflict_allows_after_choch(self) -> None:
        indicators = {
            "close": 4434.0,
            "atr": 14.0,
            "rsi": 76.0,
            "adx": 30.0,
            "market_state": "trend",
            "trend": "bullish",
            "donchian_high": 4460.0,
            "donchian_low": 4300.0,
            "structure": {
                "trend": "bearish",
                "choch_up": False,
                "choch_down": True,
                "health": 48,
                "health_state": "HOLDING",
            },
        }
        with mock.patch("tools.structure.ENABLE_TREND_CONFLICT_GUARD", True):
            guard = trend_conflict_guard("SELL", indicators)
        self.assertIsNone(guard)

    def test_trend_conflict_blocks_buy_when_structure_still_bearish(self) -> None:
        indicators = {
            "close": 4434.0,
            "atr": 14.0,
            "rsi": 60.0,
            "adx": 30.0,
            "market_state": "trend",
            "trend": "bullish",
            "donchian_high": 4460.0,
            "donchian_low": 4300.0,
            "structure": {
                "trend": "bearish",
                "choch_up": False,
                "choch_down": False,
                "health": 60,
                "health_state": "HOLDING",
            },
        }
        with mock.patch("tools.structure.ENABLE_TREND_CONFLICT_GUARD", True):
            guard = trend_conflict_guard("BUY", indicators)
        self.assertIsNotNone(guard)
        self.assertTrue(guard["block"])
        self.assertIn("趋势冲突", guard["reason"])

    def test_trend_conflict_allows_buy_after_close_through_high(self) -> None:
        indicators = {
            "close": 4462.0,
            "atr": 14.0,
            "rsi": 66.0,
            "adx": 30.0,
            "market_state": "trend",
            "trend": "bullish",
            "donchian_high": 4460.0,
            "donchian_low": 4300.0,
            "structure": {
                "trend": "bearish",
                "choch_up": False,
                "choch_down": False,
                "health": 60,
                "health_state": "HOLDING",
            },
        }
        with mock.patch("tools.structure.ENABLE_TREND_CONFLICT_GUARD", True):
            guard = trend_conflict_guard("BUY", indicators)
        self.assertIsNone(guard)

    def test_rule_veto_blocks_sell_when_bullish_votes_dominate(self) -> None:
        trader = TraderAgent(MockLLM())
        context = {
            "market_state": "trend",
            "trend": "bullish",
            "momentum_ok": True,
            "range_signal": "none",
            "close": 4434.0,
            "atr": 14.0,
            "positions": [],
            "structure": {"trend": "bearish", "health": 48},
            "momentum": {
                "latest": {"direction": "BUY", "confirm_age": 2}
            },
            "cci": {"latest": {"direction": "BUY", "confirm_age": 2}},
            "macd": {"state": "strong_bullish", "latest_trigger": None},
            "ict": {
                "fvg": {"direction": "BUY", "age": 1},
                "sweep": None,
            },
            "key_levels": {"support": None, "resistance": None},
        }
        indicators = {
            "close": 4434.0,
            "atr": 14.0,
            "two_b_bottom": False,
            "two_b_top": False,
            **context,
        }
        with mock.patch("agent.trader_agent.ENABLE_RULE_VETO", True):
            veto = trader._rule_veto(context, "SELL", indicators)
        self.assertIsNotNone(veto)
        self.assertIn("规则复核否决", veto["reason"])


class ReversalSignalTest(unittest.TestCase):
    def test_bottom_two_b_returns_buy(self) -> None:
        signal = reversal_signal(
            {
                "close": 4576.0,
                "atr": 18.0,
                "rsi": 25.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": True,
                "two_b_top": False,
            }
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "BUY")

    def test_top_two_b_returns_sell(self) -> None:
        signal = reversal_signal(
            {
                "close": 4619.0,
                "atr": 18.0,
                "rsi": 70.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": False,
                "two_b_top": True,
            }
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "SELL")

    def test_no_two_b_returns_none(self) -> None:
        signal = reversal_signal(
            {
                "close": 4600.0,
                "atr": 18.0,
                "rsi": 50.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": False,
                "two_b_top": False,
            }
        )
        self.assertIsNone(signal)

    def test_bottom_two_b_confirmed_after_reclaim_returns_buy(self) -> None:
        signal = reversal_signal(
            {
                "close": 4590.0,
                "atr": 18.0,
                "rsi": 26.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": True,
                "two_b_top": False,
            }
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "BUY")

    def test_two_b_too_far_away_is_not_confirmed(self) -> None:
        signal = reversal_signal(
            {
                "close": 4616.0,
                "atr": 18.0,
                "rsi": 26.0,
                "donchian_high": 4620.0,
                "donchian_low": 4575.0,
                "two_b_bottom": True,
                "two_b_top": False,
            }
        )
        self.assertIsNone(signal)


class SwingReversalSignalTest(unittest.TestCase):
    def test_bottom_swing_confirmation_returns_buy(self) -> None:
        closes = [
            4400.0,
            4395.0,
            4390.0,
            4380.0,
            4370.0,
            4360.0,
            4300.0,
            4305.0,
            4300.0,
            4315.0,
            4320.0,
            4320.0,
            4320.0,
            4340.0,
        ]
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-09-01", periods=len(closes), freq="h"),
                "open": closes,
                "high": [v + 8.0 for v in closes],
                "low": [v - 8.0 for v in closes],
                "close": closes,
            }
        )
        signal = swing_reversal_signal(df, {"atr": 20.0})
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "BUY")


class VisionReversalGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            mock.patch("tools.structure.ENABLE_VISION_GUARD", True),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()

    def test_stop_candle_blocks_sell_but_allows_buy(self) -> None:
        vision = {
            "structure": "H1止跌K线（长下影/阳包阴）",
            "setup": "停止追空，观察企稳",
            "confidence": 74,
        }
        sell_guard = vision_reversal_guard("SELL", vision)
        self.assertIsNotNone(sell_guard)
        self.assertIn("禁止继续追空", sell_guard["reason"])
        self.assertIsNone(vision_reversal_guard("BUY", vision))

    def test_stall_candle_blocks_buy_but_allows_sell(self) -> None:
        vision = {
            "structure": "H1滞涨K线（长上影/阴包阴）",
            "setup": "停止追多，观察回落",
            "confidence": 74,
        }
        buy_guard = vision_reversal_guard("BUY", vision)
        self.assertIsNotNone(buy_guard)
        self.assertIn("禁止继续追多", buy_guard["reason"])
        self.assertIsNone(vision_reversal_guard("SELL", vision))


class ExtremeChaseGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            mock.patch("tools.structure.ENABLE_EXTREME_GUARD", True),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()

    def test_blocks_extended_short_into_oversold_low(self) -> None:
        guard = extreme_chase_guard(
            "SELL",
            {
                "close": 4315.0,
                "atr": 20.0,
                "rsi": 25.0,
                "market_state": "trend",
                "trend": "bearish",
                "donchian_high": 4390.0,
                "donchian_low": 4300.0,
                "closes": [4390.0, 4380.0, 4360.0, 4345.0, 4330.0, 4315.0],
            },
        )
        self.assertIsNotNone(guard)
        self.assertIn("禁止继续追空", guard["reason"])

    def test_allows_mid_trend_short(self) -> None:
        guard = extreme_chase_guard(
            "SELL",
            {
                "close": 4420.0,
                "atr": 20.0,
                "rsi": 45.0,
                "market_state": "trend",
                "trend": "bearish",
                "donchian_high": 4520.0,
                "donchian_low": 4300.0,
                "closes": [4460.0, 4450.0, 4440.0, 4430.0, 4425.0, 4420.0],
            },
        )
        self.assertIsNone(guard)


class EntryQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            mock.patch("tools.quality.ENABLE_QUALITY_GATE", True),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()

    @staticmethod
    def _base(action: str) -> dict:
        htf_trend = "bullish" if action == "BUY" else "bearish"
        return {
            "htf_trend": htf_trend,
            "htf_adx": 25.0,
            "htf_rsi": 55.0 if action == "BUY" else 45.0,
        }

    def test_blocks_trend_buy_with_weak_adx(self) -> None:
        block = entry_quality(
            "BUY",
            {
                **self._base("BUY"),
                "close": 4600.0,
                "rsi": 55.0,
                "atr": 18.0,
                "adx": 15.0,
                "donchian_high": 4640.0,
                "donchian_low": 4570.0,
                "market_state": "trend",
                "trend": "bullish",
                "two_b_bottom": False,
                "two_b_top": False,
            },
        )
        self.assertIsNotNone(block)
        self.assertIn("ADX", block["reason"])

    def test_allows_trend_buy_with_strong_adx(self) -> None:
        block = entry_quality(
            "BUY",
            {
                **self._base("BUY"),
                "close": 4600.0,
                "rsi": 55.0,
                "atr": 18.0,
                "adx": 28.0,
                "donchian_high": 4640.0,
                "donchian_low": 4570.0,
                "market_state": "trend",
                "trend": "bullish",
                "two_b_bottom": False,
                "two_b_top": False,
            },
        )
        self.assertIsNone(block)

    def test_blocks_range_buy_away_from_bottom(self) -> None:
        block = entry_quality(
            "BUY",
            {
                **self._base("BUY"),
                "close": 4610.0,
                "rsi": 40.0,
                "atr": 18.0,
                "adx": 18.0,
                "donchian_high": 4640.0,
                "donchian_low": 4570.0,
                "market_state": "range",
                "trend": "neutral",
                "two_b_bottom": False,
                "two_b_top": False,
            },
        )
        self.assertIsNotNone(block)

    def test_allows_confirmed_two_b_reversal(self) -> None:
        block = entry_quality(
            "SELL",
            {
                **self._base("SELL"),
                "close": 4636.0,
                "rsi": 72.0,
                "atr": 18.0,
                "adx": 15.0,
                "donchian_high": 4640.0,
                "donchian_low": 4570.0,
                "market_state": "trend",
                "trend": "bullish",
                "two_b_bottom": False,
                "two_b_top": True,
            },
        )
        self.assertIsNone(block)

    def test_blocks_buy_when_htf_bearish(self) -> None:
        indicators = {
            **self._base("SELL"),
            "close": 4600.0,
            "rsi": 55.0,
            "atr": 18.0,
            "adx": 30.0,
            "donchian_high": 4640.0,
            "donchian_low": 4570.0,
            "market_state": "trend",
            "trend": "bullish",
            "two_b_bottom": False,
            "two_b_top": False,
        }
        with mock.patch("tools.quality.ENABLE_HTF_FILTER", True):
            self.assertIn(
                "大周期",
                entry_quality("BUY", indicators)["reason"],
            )
        with mock.patch("tools.quality.ENABLE_HTF_FILTER", False):
            self.assertIsNone(entry_quality("BUY", indicators))

    def test_htf_requires_strength_confluence(self) -> None:
        weak_adx = {
            **self._base("BUY"),
            "htf_adx": 10.0,
            "close": 4600.0,
            "rsi": 55.0,
            "atr": 18.0,
            "adx": 30.0,
            "donchian_high": 4640.0,
            "donchian_low": 4570.0,
            "market_state": "trend",
            "trend": "bullish",
            "two_b_bottom": False,
            "two_b_top": False,
        }
        weak_rsi = {
            **weak_adx,
            "htf_adx": 25.0,
            "htf_rsi": 40.0,
        }
        with mock.patch("tools.quality.ENABLE_HTF_FILTER", True):
            self.assertIsNotNone(entry_quality("BUY", weak_adx))
            self.assertIsNotNone(entry_quality("BUY", weak_rsi))


class CloseSignalTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            mock.patch("tools.structure.ENABLE_AI_CLOSE", True),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()

    def test_requires_all_confirmations(self) -> None:
        indicators = {
            "close": 4576.44,
            "atr": 18.0,
            "rsi": 40.0,
            "donchian_high": 4620.0,
            "donchian_low": 4575.0,
            "two_b_bottom": True,
            "two_b_top": False,
        }
        signal = close_signal(indicators, [{"side": "SELL", "entry": 4590.0}])
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "CLOSE")
        # 浮盈持仓在趋势反转确认时同样平仓。
        profitable = close_signal(
            indicators, [{"side": "SELL", "entry": 4590.0, "profit": 50.0}]
        )
        self.assertIsNotNone(profitable)
        # Missing RSI direction or 2B removes the signal.
        indicators_no_2b = {**indicators, "two_b_bottom": False}
        self.assertIsNone(
            close_signal(indicators_no_2b, [{"side": "SELL", "entry": 4590.0}])
        )
        indicators_no_rsi = {**indicators, "rsi": 60.0}
        self.assertIsNone(
            close_signal(indicators_no_rsi, [{"side": "SELL", "entry": 4590.0}])
        )

    def test_no_positions_returns_none(self) -> None:
        indicators = {
            "close": 4576.44,
            "atr": 18.0,
            "rsi": 40.0,
            "donchian_high": 4620.0,
            "donchian_low": 4575.0,
            "two_b_bottom": True,
            "two_b_top": False,
        }
        self.assertIsNone(close_signal(indicators, []))

    def test_trend_conflict_sell_close(self) -> None:
        indicators = {
            "close": 4604.78,
            "sma20": 4598.63,
            "atr": 7.54,
            "rsi": 64.5,
            "donchian_high": 4614.56,
            "donchian_low": 4571.70,
            "two_b_bottom": False,
            "two_b_top": False,
            "market_state": "trend",
            "trend": "bullish",
            "momentum_ok": True,
        }
        signal = close_signal(
            indicators,
            [{"side": "SELL", "entry": 4612.52, "profit": 7.37}],
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "CLOSE")
        no_rsi = {**indicators, "rsi": 50.0}
        self.assertIsNone(
            close_signal(no_rsi, [{"side": "SELL", "entry": 4612.52}])
        )


class _FakeAccount:
    def __init__(self, positions: list[dict]) -> None:
        self._positions = positions

    def snapshot(self) -> dict:
        return {
            "balance": 10000,
            "equity": 10000,
            "open_positions": len(self._positions),
            "daily_pnl": 0.0,
            "loss_streak": 0,
            "loss_paused": False,
            "positions": self._positions,
        }


class RiskManagerAddonTest(unittest.TestCase):
    def test_blocks_new_entry_when_position_open(self) -> None:
        risk = RiskManager(
            None,
            account=_FakeAccount([{"side": "BUY", "volume": 0.01, "entry": 3350}]),
        )
        review = risk.review(
            {
                "action": "BUY",
                "confidence": 72,
                "risk_percent": 0.01,
                "market_state": "趋势",
                "add_on": False,
            },
            {
                "close": 3350,
                "atr": 4,
                "indicators": {"market_state": "trend", "adx": 30},
            },
        )
        self.assertFalse(review["allow"])

    def test_allows_new_entry_on_different_symbol(self) -> None:
        risk = RiskManager(
            None,
            account=_FakeAccount(
                [{"side": "SELL", "symbol": "EURUSD", "volume": 0.01, "entry": 1.08}]
            ),
        )
        review = risk.review(
            {
                "action": "BUY",
                "confidence": 85,
                "risk_percent": 0.01,
                "market_state": "趋势",
                "add_on": False,
            },
            {
                "symbol": "XAUUSD",
                "close": 3350,
                "atr": 4,
                "indicators": {"market_state": "trend", "adx": 30},
            },
        )
        self.assertTrue(review["allow"])

    def test_blocks_second_add_on_on_same_symbol(self) -> None:
        risk = RiskManager(
            None,
            account=_FakeAccount(
                [
                    {"side": "BUY", "symbol": "XAUUSD", "volume": 0.01, "entry": 3350},
                    {"side": "BUY", "symbol": "XAUUSD", "volume": 0.01, "entry": 3355},
                ]
            ),
        )
        review = risk.review(
            {
                "action": "BUY",
                "confidence": 90,
                "risk_percent": 0.01,
                "market_state": "趋势",
                "add_on": True,
            },
            {
                "symbol": "XAUUSD",
                "close": 3360,
                "atr": 4,
                "indicators": {"market_state": "trend", "adx": 40},
            },
        )
        self.assertFalse(review["allow"])
        self.assertTrue(
            any("加仓次数已达上限" in reason for reason in review["reasons"])
        )

    def test_allows_strong_trend_addon(self) -> None:
        risk = RiskManager(
            None,
            account=_FakeAccount([{"side": "BUY", "volume": 0.01, "entry": 3350}]),
        )
        review = risk.review(
            {
                "action": "BUY",
                "confidence": 82,
                "risk_percent": 0.01,
                "market_state": "趋势",
                "add_on": True,
            },
            {
                "close": 3350,
                "atr": 4,
                "indicators": {"market_state": "trend", "adx": 35},
            },
        )
        self.assertTrue(review["allow"])

    def test_blocks_opposite_side_addon(self) -> None:
        risk = RiskManager(
            None,
            account=_FakeAccount([{"side": "SELL", "volume": 0.01, "entry": 3350}]),
        )
        review = risk.review(
            {
                "action": "BUY",
                "confidence": 85,
                "risk_percent": 0.01,
                "market_state": "趋势",
                "add_on": True,
            },
            {
                "close": 3350,
                "atr": 4,
                "indicators": {"market_state": "trend", "adx": 35},
            },
        )
        self.assertFalse(review["allow"])

    def test_blocks_addon_when_adx_weak(self) -> None:
        risk = RiskManager(
            None,
            account=_FakeAccount([{"side": "BUY", "volume": 0.01, "entry": 3350}]),
        )
        review = risk.review(
            {
                "action": "BUY",
                "confidence": 85,
                "risk_percent": 0.01,
                "market_state": "趋势",
                "add_on": True,
            },
            {
                "close": 3350,
                "atr": 4,
                "indicators": {"market_state": "trend", "adx": 20},
            },
        )
        self.assertFalse(review["allow"])


class RiskManagerSmallAccountGuardTest(unittest.TestCase):
    class _Account:
        def __init__(self, balance: float) -> None:
            self._balance = balance

        def snapshot(self) -> dict:
            balance = float(self._balance)
            return {
                "balance": balance,
                "equity": balance,
                "open_positions": 0,
                "daily_pnl": 0.0,
                "loss_streak": 0,
                "loss_paused": False,
                "positions": [],
            }

    def _review(self, balance: float) -> dict:
        risk = RiskManager(None, account=self._Account(balance))
        return risk.review(
            {
                "action": "BUY",
                "confidence": 85,
                "risk_percent": 0.01,
                "sl": 4380.0,
                "tp": 4420.0,
            },
            {
                "close": 4400.0,
                "atr": 20.0,
                "symbol": "XAUUSD",
                "indicators": {"market_state": "trend", "adx": 30},
            },
            positions=[],
        )

    def test_blocks_small_account_min_lot_over_risk(self) -> None:
        with mock.patch("risk.manager.ENABLE_MIN_LOT_RISK_GUARD", True):
            review = self._review(200.0)
        self.assertFalse(review["allow"])
        self.assertTrue(
            any("0.01手最小仓" in reason for reason in review["reasons"])
        )

    def test_allows_when_min_lot_risk_fits_limit(self) -> None:
        with mock.patch("risk.manager.ENABLE_MIN_LOT_RISK_GUARD", True):
            review = self._review(20000.0)
        self.assertTrue(review["allow"])

    def test_can_be_disabled(self) -> None:
        with mock.patch("risk.manager.ENABLE_MIN_LOT_RISK_GUARD", False):
            review = self._review(200.0)
        self.assertTrue(review["allow"])


class RiskManagerStopWidthGuardTest(unittest.TestCase):
    class _Account:
        def snapshot(self) -> dict:
            return {
                "balance": 10000.0,
                "equity": 10000.0,
                "open_positions": 0,
                "daily_pnl": 0.0,
                "loss_streak": 0,
                "loss_paused": False,
                "positions": [],
            }

    def _review(self, sl: float) -> dict:
        risk = RiskManager(None, account=self._Account())
        return risk.review(
            {
                "action": "BUY",
                "confidence": 85,
                "risk_percent": 0.01,
                "sl": sl,
                "tp": 102.0,
            },
            {
                "symbol": "XAUUSD",
                "close": 100.0,
                "atr": 2.0,
                "indicators": {"market_state": "trend", "adx": 30},
            },
            positions=[],
        )

    def test_blocks_ultra_tight_stop(self) -> None:
        with mock.patch("risk.manager.ENABLE_MIN_STOP_ATR_GUARD", True):
            review = self._review(99.95)
        self.assertFalse(review["allow"])
        self.assertTrue(
            any("止损距离过近" in reason for reason in review["reasons"])
        )

    def test_allows_atr_anchored_stop(self) -> None:
        with mock.patch("risk.manager.ENABLE_MIN_STOP_ATR_GUARD", True):
            review = self._review(98.0)
        self.assertTrue(review["allow"])


class RiskManagerHardPauseTest(unittest.TestCase):
    @staticmethod
    def _risk(pnl: float, paused: bool = False) -> RiskManager:
        class _Account:
            def snapshot(self) -> dict:
                return {
                    "balance": 10000.0,
                    "daily_pnl": pnl,
                    "loss_paused": paused,
                }

        return RiskManager(None, account=_Account())

    def test_daily_loss_lock_returns_reason(self) -> None:
        with mock.patch("risk.manager.ENABLE_DAILY_LOSS_LIMIT", True), mock.patch(
            "risk.manager.DAILY_LOSS_LIMIT", 0.03
        ):
            reason = self._risk(-400.0).hard_pause_reason()
        self.assertEqual(reason, "daily loss limit reached")

    def test_loss_pause_lock_returns_reason(self) -> None:
        with mock.patch("risk.manager.ENABLE_LOSS_PAUSE", True):
            reason = self._risk(0.0, paused=True).hard_pause_reason()
        self.assertIn("连亏暂停", reason)


class OrderManagerTest(unittest.TestCase):
    def test_builds_sized_order(self) -> None:
        manager = OrderManager()
        order = manager.build(
            {
                "action": "BUY",
                "confidence": 80,
                "risk_percent": 0.01,
                "sl": 3330,
                "tp": 3390,
            },
            {"close": 3350, "atr": 4, "symbol": "XAUUSD"},
            {"balance": 10000},
        )
        self.assertIsNotNone(order)
        self.assertEqual(order["action"], "BUY")
        self.assertGreater(order["volume"], 0)
        self.assertLess(order["sl"], order["entry"])
        self.assertGreater(order["tp"], order["entry"])

    def test_builds_sized_forex_order(self) -> None:
        manager = OrderManager()
        order = manager.build(
            {
                "action": "BUY",
                "confidence": 80,
                "risk_percent": 0.01,
                "sl": 1.085,
                "tp": 1.095,
            },
            {"close": 1.09, "atr": 0.003, "symbol": "EURUSD"},
            {"balance": 10000},
        )
        self.assertIsNotNone(order)
        self.assertAlmostEqual(float(order["volume"]), 0.2, places=2)

    def test_rejects_unknown_symbol(self) -> None:
        manager = OrderManager()
        with self.assertRaises(ValueError):
            manager.build(
                {
                    "action": "BUY",
                    "confidence": 80,
                    "risk_percent": 0.01,
                    "sl": 1.0,
                    "tp": 2.0,
                },
                {"close": 1.5, "atr": 0.1, "symbol": "UNKNOWN:PAIR"},
                {"balance": 10000},
            )

    def test_builds_close_order(self) -> None:
        manager = OrderManager()
        order = manager.build(
            {"action": "CLOSE", "confidence": 80, "risk_percent": 0.01},
            {"close": 4576.44, "atr": 18.0, "symbol": "XAUUSD"},
            {"balance": 10000},
        )
        self.assertIsNotNone(order)
        self.assertEqual(order["action"], "CLOSE")
        self.assertNotIn("sl", order)
        self.assertNotIn("volume", order)


class PaperBrokerTest(DatabaseTestCase):
    def test_full_position_lifecycle(self) -> None:
        broker = PaperBroker(self.db)
        broker.place(
            {
                "action": "BUY",
                "symbol": "XAUUSD",
                "entry": 100.0,
                "sl": 99.0,
                "tp": 103.0,
                "volume": 1.0,
                "magic": 1,
                "reason": "test",
                "confidence": 80,
            }
        )
        events: list[dict] = []
        for price in (100.0, 100.5, 101.5, 102.0, 105.0):
            events.extend(broker.tick(price))
        event_types = [event["type"] for event in events]
        self.assertIn("BREAKEVEN", event_types)
        self.assertIn("PARTIAL_TP", event_types)
        self.assertIn("TRAILING", event_types)
        self.assertIn("CLOSE_TP", event_types)

        closed = self.db.closed_trades()
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["status"], "closed")
        # 225 gross minus 0.25 spread cost on 1.0 lot (=25).
        self.assertAlmostEqual(float(closed[0]["pnl"]), 200.0, places=2)

    def test_min_lot_does_not_partial_close_entire_position(self) -> None:
        broker = PaperBroker(self.db)
        broker.place(
            {
                "action": "BUY",
                "symbol": "XAUUSD",
                "entry": 100.0,
                "sl": 90.0,
                "tp": 110.0,
                "volume": 0.01,
                "magic": 1,
                "reason": "min lot partial test",
                "confidence": 80,
            }
        )
        events = broker.tick(105.0)
        self.assertNotIn(
            "PARTIAL_TP", [event["type"] for event in events]
        )
        open_trade = self.db.open_trades()[0]
        self.assertEqual(open_trade["partial_done"], 0)
        self.assertEqual(open_trade["volume"], 0.01)

    def test_two_lots_still_partial_close_half(self) -> None:
        broker = PaperBroker(self.db)
        broker.place(
            {
                "action": "BUY",
                "symbol": "XAUUSD",
                "entry": 100.0,
                "sl": 90.0,
                "tp": 110.0,
                "volume": 0.02,
                "magic": 1,
                "reason": "half lot partial test",
                "confidence": 80,
            }
        )
        events = broker.tick(105.0)
        event_types = [event["type"] for event in events]
        self.assertIn("PARTIAL_TP", event_types)
        open_trade = self.db.open_trades()[0]
        self.assertEqual(open_trade["partial_done"], 1)


class IndicatorsTest(unittest.TestCase):
    def test_computes_trend_rsi_atr(self) -> None:
        closes = [float(v) for v in range(3200, 3320)]
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=len(closes), freq="h"),
                "open": closes,
                "high": [c + 2 for c in closes],
                "low": [c - 2 for c in closes],
                "close": closes,
                "tick_volume": [100] * len(closes),
            }
        )
        indicators = compute_indicators(df)
        self.assertEqual(indicators["trend"], "bullish")
        self.assertGreater(indicators["atr"], 0)
        self.assertGreaterEqual(indicators["rsi"], 0)
        self.assertLessEqual(indicators["rsi"], 100)

    def test_detects_two_b_bottom(self) -> None:
        n = 26
        closes = [100.0 + (i % 6) * 1.5 for i in range(n)]
        highs = [c + 3.0 for c in closes]
        lows = [c - 3.0 for c in closes]
        # Previous bar pierces the 20-bar support, current bar fails to make a
        # new low and closes back above it: classic bullish 2B.
        closes[-2] = 99.0
        highs[-2] = 99.5
        lows[-2] = 94.0
        closes[-1] = 100.5
        highs[-1] = 101.0
        lows[-1] = 95.0
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
                "open": closes,
                "high": highs,
                "low": lows,
                "close": closes,
                "tick_volume": [100] * n,
            }
        )
        indicators = compute_indicators(df)
        self.assertTrue(indicators["two_b_bottom"])
        self.assertFalse(indicators["two_b_top"])

    def test_detects_two_b_top(self) -> None:
        n = 26
        closes = [100.0 + (i % 6) * 1.0 for i in range(n)]
        highs = [c + 2.0 for c in closes]
        lows = [c - 2.0 for c in closes]
        closes[-2] = 104.0
        highs[-2] = 109.0
        lows[-2] = 103.5
        closes[-1] = 102.5
        highs[-1] = 108.0
        lows[-1] = 102.0
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
                "open": closes,
                "high": highs,
                "low": lows,
                "close": closes,
                "tick_volume": [100] * n,
            }
        )
        indicators = compute_indicators(df)
        self.assertTrue(indicators["two_b_top"])
        self.assertFalse(indicators["two_b_bottom"])


class MarketStructureTest(unittest.TestCase):
    @staticmethod
    def _wave_df(points, per_seg=28, bars=None) -> pd.DataFrame:
        prices: list[float] = []
        for index in range(len(points) - 1):
            start, end = points[index], points[index + 1]
            for step in range(per_seg):
                prices.append(start + (end - start) * (step + 1) / per_seg)
        if bars is not None:
            prices = prices[:bars]
        frame = pd.DataFrame(
            {
                "time": pd.date_range(
                    "2026-01-01", periods=len(prices), freq="h"
                ),
                "open": prices,
                "high": [price + 3.0 for price in prices],
                "low": [price - 3.0 for price in prices],
                "close": prices,
                "tick_volume": [100] * len(prices),
            }
        )
        return frame

    def test_confirmed_choch_flips_structure(self) -> None:
        df = self._wave_df([100, 130, 110, 150, 120, 90, 175], bars=161)
        snapshot = structure_snapshot(df, swing_len=10, atr=6.0)
        self.assertEqual(snapshot["trend"], "bullish")
        self.assertTrue(snapshot["choch_up"])
        self.assertFalse(snapshot["choch_down"])
        self.assertFalse(snapshot["bos_up"])
        self.assertEqual(snapshot["tape"][-1], "CHoCH up")
        # A fresh CHoCH has no second leg to measure, so health stays None.
        self.assertIsNone(snapshot["health"])
        self.assertEqual(snapshot["health_state"], "NEW_TREND")
        text = structure_reference_text(snapshot)
        self.assertIn("CHoCH", text)
        self.assertIn("多头", text)

    def test_compute_indicators_exposes_structure(self) -> None:
        closes = [100.0 + 12.0 * math.sin(index / 7.0) + index * 0.12 for index in range(120)]
        df = pd.DataFrame(
            {
                "time": pd.date_range(
                    "2026-01-01", periods=len(closes), freq="h"
                ),
                "open": closes,
                "high": [value + 2.5 for value in closes],
                "low": [value - 2.5 for value in closes],
                "close": closes,
                "tick_volume": [100] * len(closes),
            }
        )
        indicators = compute_indicators(df)
        snapshot = indicators["structure"]
        self.assertIn(snapshot["trend"], {"bullish", "bearish", "neutral"})
        self.assertIn("swing_high", snapshot)
        self.assertIn("swing_low", snapshot)
        self.assertIsInstance(snapshot["tape"], list)
        if snapshot["health"] is not None:
            self.assertGreaterEqual(snapshot["health"], 0)
            self.assertLessEqual(snapshot["health"], 100)

    def test_short_frame_is_neutral(self) -> None:
        df = self._wave_df([100, 110], per_seg=4)
        snapshot = structure_snapshot(df, swing_len=10)
        self.assertEqual(snapshot["trend"], "neutral")
        self.assertEqual(snapshot["tape"], [])
        self.assertIsNone(snapshot["health"])


class MarketContextTest(unittest.TestCase):
    @staticmethod
    def _wave_df(points, per_seg=28, bars=None) -> pd.DataFrame:
        prices: list[float] = []
        for index in range(len(points) - 1):
            start, end = points[index], points[index + 1]
            for step in range(per_seg):
                prices.append(start + (end - start) * (step + 1) / per_seg)
        if bars is not None:
            prices = prices[:bars]
        return pd.DataFrame(
            {
                "time": pd.date_range(
                    "2026-01-01", periods=len(prices), freq="h"
                ),
                "open": prices,
                "high": [price + 3.0 for price in prices],
                "low": [price - 3.0 for price in prices],
                "close": prices,
                "tick_volume": [100] * len(prices),
            }
        )

    def test_indicators_expose_momentum_and_key_levels(self) -> None:
        df = self._wave_df([100, 130, 110, 150, 120, 90, 175])
        indicators = compute_indicators(df)
        self.assertIn("momentum", indicators)
        self.assertIn("key_levels", indicators)
        self.assertIn("momentum_reference", indicators)
        self.assertIn("key_level_reference", indicators)
        self.assertIn("cci", indicators)
        self.assertIn("cci_reference", indicators)
        self.assertIn("ict", indicators)
        self.assertIn("ict_reference", indicators)
        self.assertIn("macd", indicators)
        self.assertIn("macd_reference", indicators)
        self.assertIn("trend_phase", indicators)
        self.assertIn("trend_phase_reference", indicators)

    def test_short_momentum_frame_is_empty(self) -> None:
        df = self._wave_df([100, 110], per_seg=4)
        snapshot = momentum_snapshot(df, rsi_pivot_len=10)
        self.assertFalse(snapshot["regular_bullish"])
        self.assertFalse(snapshot["regular_bearish"])
        self.assertIsNone(snapshot["latest"])
        self.assertEqual(momentum_reference_text(snapshot), "暂无已确认的动量背离")

    def test_key_level_snapshot_returns_tested_zones(self) -> None:
        df = self._wave_df([100, 130, 110, 150, 120, 90, 175])
        snapshot = key_level_snapshot(df, swing_len=10)
        self.assertGreaterEqual(len(snapshot["zones"]), 1)
        self.assertIn("support", snapshot)
        self.assertIn("resistance", snapshot)
        text = key_level_reference_text(snapshot)
        self.assertTrue(("支撑" in text) or ("阻力" in text))

    def test_cci_short_frame_is_empty(self) -> None:
        df = self._wave_df([100, 110], per_seg=4)
        snapshot = cci_divergence_snapshot(df)
        self.assertFalse(snapshot["bullish"])
        self.assertFalse(snapshot["bearish"])
        self.assertIsNone(snapshot["latest"])
        self.assertIn("暂无", cci_reference_text(snapshot))

    def test_ict_snapshot_is_context_only(self) -> None:
        df = self._wave_df([100, 130, 110, 150, 120, 90, 175])
        snapshot = ict_context_snapshot(df, sweep_lookback=20, fvg_lookback=30)
        self.assertIn("sweep", snapshot)
        self.assertIn("fvg", snapshot)
        text = ict_reference_text(snapshot)
        self.assertIsInstance(text, str)

    def test_macd_short_frame_is_neutral(self) -> None:
        df = self._wave_df([100, 110], per_seg=4)
        snapshot = macd_momentum_snapshot(df)
        self.assertEqual(snapshot["state"], "neutral")
        self.assertIsNone(snapshot["latest_trigger"])
        self.assertIn("MACD", macd_reference_text(snapshot))

    def test_trend_phase_flags_exhaustion_risk(self) -> None:
        indicators = {
            "close": 4428.0,
            "atr": 18.0,
            "rsi": 25.0,
            "adx": 40.0,
            "market_state": "trend",
            "trend": "bearish",
            "donchian_high": 4460.0,
            "donchian_low": 4300.0,
            "structure": {
                "trend": "bearish",
                "health": 38,
                "health_state": "WEAKENING",
                "swing_high": 4460.0,
                "swing_low": 4280.0,
                "tape": ["BOS down", "BOS down"],
            },
            "momentum": {
                "latest": {
                    "type": "regular_bullish",
                    "direction": "BUY",
                    "confirm_age": 5,
                }
            },
            "cci": {"latest": None},
            "ict": {},
        }
        phase = trend_phase_snapshot(indicators)
        self.assertEqual(phase["phase"], "exhaustion_risk")
        self.assertIn("末端风险", trend_phase_reference_text(phase))

    def test_trend_phase_flags_structure_flip(self) -> None:
        indicators = {
            "close": 4400.0,
            "atr": 18.0,
            "rsi": 50.0,
            "adx": 20.0,
            "market_state": "range",
            "trend": "neutral",
            "donchian_high": 4460.0,
            "donchian_low": 4300.0,
            "structure": {
                "trend": "bullish",
                "health": None,
                "health_state": "NEW_TREND",
                "swing_high": 4460.0,
                "swing_low": 4300.0,
                "tape": ["BOS down", "CHoCH up"],
            },
            "momentum": {"latest": None},
            "cci": {"latest": None},
            "ict": {},
        }
        phase = trend_phase_snapshot(indicators)
        self.assertEqual(phase["phase"], "structure_flip")

    def test_position_state_detects_wake_change(self) -> None:
        before = [
            {"side": "BUY", "ticket": 101, "symbol": "XAUUSD"},
            {"side": "SELL", "ticket": 102, "symbol": "XAUUSD"},
        ]
        after = [
            {"side": "BUY", "ticket": 101, "symbol": "XAUUSD"},
        ]
        self.assertEqual(
            TradingCEO._position_state(before),
            (
                ("BUY", "101", "XAUUSD"),
                ("SELL", "102", "XAUUSD"),
            ),
        )
        self.assertNotEqual(
            TradingCEO._position_state(before),
            TradingCEO._position_state(after),
        )

    def test_feature_brain_requires_optional_vote(self) -> None:
        ctx = {
            "market_state": "trend",
            "trend": "bullish",
            "momentum_ok": True,
            "range_signal": "none",
            "close": 4430.0,
            "atr": 14.0,
            "positions": [],
            "structure": {"trend": "bullish", "health": 70},
            "momentum": {"latest": {"direction": "BUY", "confirm_age": 2}},
            "cci": {"latest": None},
            "macd": {"state": "strong_bullish", "latest_trigger": None},
            "ict": {"fvg": None, "sweep": None},
            "key_levels": {"support": None, "resistance": None},
        }
        brain = FeatureBrain("base,structure,rsi,cci,macd,ict,key")
        self.assertEqual(brain.decide(ctx)["action"], "BUY")
        blocked = FeatureBrain("base,key")
        self.assertEqual(blocked.decide(ctx)["action"], "WAIT")


class StrategyAdvisorTest(unittest.TestCase):
    def test_trend_prefers_momentum(self) -> None:
        advice = StrategyAdvisor().advise(
            {"indicators": {"market_state": "trend"}}
        )
        self.assertEqual(advice["preferred_persona"], "momentum")

    def test_range_prefers_mean_reversion(self) -> None:
        advice = StrategyAdvisor().advise(
            {"indicators": {"market_state": "range"}}
        )
        self.assertEqual(advice["preferred_persona"], "mean_reversion")


class RiskNormalizationTest(unittest.TestCase):
    def test_percent_style_risk(self) -> None:
        self.assertEqual(normalize_risk(1.0), 0.01)
        self.assertEqual(normalize_risk(0.01), 0.01)


class StateMachineRecoveryTest(unittest.TestCase):
    def test_executing_can_return_to_waiting_after_order_failure(self) -> None:
        machine = StateMachine()
        machine.transition(State.ANALYZING)
        machine.transition(State.PLANNING)
        machine.transition(State.RISK_CHECK)
        machine.transition(State.EXECUTING)
        machine.transition(State.WAITING)
        self.assertEqual(machine.state, State.WAITING)


class AccountTimestampTest(unittest.TestCase):
    def test_parse_aware_utc_timestamp(self) -> None:
        from mt5.account import Account

        parsed = Account._parse_ts("2026-08-28T10:00:00Z")
        self.assertEqual(
            parsed,
            datetime(2026, 8, 28, 10, 0, 0, tzinfo=timezone.utc),
        )


class VisionCandleTest(unittest.TestCase):
    def test_detects_stop_candle_in_downtrend(self) -> None:
        from vision.analyzer import VisionAnalyzer

        bars = []
        for idx in range(8):
            base = 4470.0 - idx * 8.0
            bars.append(
                {
                    "time": f"2026-09-01T{idx:02d}:00:00",
                    "open": base + 2.0,
                    "high": base + 6.0,
                    "low": base - 2.0,
                    "close": base,
                }
            )
        # 最后一根：长下影 + 收回上半区，出现在近期低点附近。
        bars.append(
            {
                "time": "2026-09-01T08:00:00",
                "open": 4412.0,
                "high": 4424.0,
                "low": 4385.0,
                "close": 4418.0,
            }
        )
        result = VisionAnalyzer._local_analysis(
            {"trend": "bearish", "bars": bars}
        )
        self.assertIn("止跌", result["structure"])
        self.assertIn("停止追空", result["setup"])

    def test_detects_stall_candle_in_uptrend(self) -> None:
        from vision.analyzer import VisionAnalyzer

        bars = []
        for idx in range(8):
            base = 4360.0 + idx * 8.0
            bars.append(
                {
                    "time": f"2026-09-01T{idx:02d}:00:00",
                    "open": base - 2.0,
                    "high": base + 3.0,
                    "low": base - 4.0,
                    "close": base,
                }
            )
        # 最后一根：长上影冲高回落，收下半区，出现在近期高点附近。
        bars.append(
            {
                "time": "2026-09-01T08:00:00",
                "open": 4418.0,
                "high": 4450.0,
                "low": 4416.0,
                "close": 4420.0,
            }
        )
        result = VisionAnalyzer._local_analysis(
            {"trend": "bullish", "bars": bars}
        )
        self.assertIn("滞涨", result["structure"])
        self.assertIn("停止追多", result["setup"])

    def test_stop_candle_stays_active_until_broken(self) -> None:
        from vision.analyzer import VisionAnalyzer

        bars = []
        for idx in range(8):
            base = 4470.0 - idx * 8.0
            bars.append(
                {
                    "time": f"2026-09-01T{idx:02d}:00:00",
                    "open": base + 2.0,
                    "high": base + 6.0,
                    "low": base - 2.0,
                    "close": base,
                }
            )
        bars.append(
            {
                "time": "2026-09-01T08:00:00",
                "open": 4412.0,
                "high": 4424.0,
                "low": 4385.0,
                "close": 4418.0,
            }
        )
        # The stop candle is still valid two bars later: no close broke the low.
        bars.append(
            {
                "time": "2026-09-01T09:00:00",
                "open": 4418.0,
                "high": 4430.0,
                "low": 4415.0,
                "close": 4425.0,
            }
        )
        bars.append(
            {
                "time": "2026-09-01T10:00:00",
                "open": 4425.0,
                "high": 4435.0,
                "low": 4420.0,
                "close": 4428.0,
            }
        )
        result = VisionAnalyzer._local_analysis(
            {"trend": "bearish", "bars": bars}
        )
        self.assertIn("止跌", result["structure"])
        self.assertIn("停止追空", result["setup"])

    def test_stop_candle_expires_when_close_breaks_low(self) -> None:
        from vision.analyzer import VisionAnalyzer

        bars = []
        for idx in range(8):
            base = 4470.0 - idx * 8.0
            bars.append(
                {
                    "time": f"2026-09-01T{idx:02d}:00:00",
                    "open": base + 2.0,
                    "high": base + 6.0,
                    "low": base - 2.0,
                    "close": base,
                }
            )
        bars.append(
            {
                "time": "2026-09-01T08:00:00",
                "open": 4412.0,
                "high": 4424.0,
                "low": 4385.0,
                "close": 4418.0,
            }
        )
        bars.append(
            {
                "time": "2026-09-01T09:00:00",
                "open": 4418.0,
                "high": 4430.0,
                "low": 4415.0,
                "close": 4425.0,
            }
        )
        # A bearish close below the stop candle low invalidates the signal.
        bars.append(
            {
                "time": "2026-09-01T10:00:00",
                "open": 4422.0,
                "high": 4424.0,
                "low": 4378.0,
                "close": 4380.0,
            }
        )
        result = VisionAnalyzer._local_analysis(
            {"trend": "bearish", "bars": bars}
        )
        text = result["structure"] + result["setup"]
        self.assertNotIn("止跌", text)
        self.assertNotIn("停止追空", text)


class NewsFilterTest(unittest.TestCase):
    def test_relevance_filters_gold_and_fed(self) -> None:
        from tools.news import _is_relevant

        self.assertTrue(_is_relevant("黄金价格短线走高，美元回落"))
        self.assertTrue(_is_relevant("FOMC 会议纪要公布"))
        self.assertFalse(_is_relevant("某地天气晴朗"))

    def test_dedupe_removes_identical_headlines(self) -> None:
        from tools.news import _dedupe

        items = [
            {"source": "a", "text": "美联储隔夜逆回购规模下降"},
            {"source": "b", "text": "美联储隔夜逆回购规模下降"},
            {"source": "c", "text": "黄金突破关键阻力"},
        ]
        result = _dedupe(items)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
