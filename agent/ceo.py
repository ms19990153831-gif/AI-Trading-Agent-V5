"""Trading CEO: orchestrates the full V5 multi-agent loop."""

from __future__ import annotations

import os
from datetime import datetime

from agent.macro_agent import MacroAgent
from agent.market_agent import MarketAgent
from agent.reflection_agent import ReflectionAgent, summarize_strategy_events
from agent.research_agent import ResearchAgent
from agent.risk_agent import RiskAgent
from agent.strategy_advisor import StrategyAdvisor
from agent.trader_agent import TraderAgent
from agent.vision_agent import VisionAgent
from brain.memory import Memory
from config import (
    DAILY_REVIEW_HOUR,
    DATA_DIR,
    ENABLE_BAR_CLOSE_DECISIONS,
    ENABLE_MACRO_GATE,
    DECISION_WAKE_ATR_MULTIPLE,
    EXECUTION_MODE,
    LITE_AGENTS,
    MID_BAR_WAKE_MODE,
    MT5_MAGIC,
    RESEARCH_INTERVAL,
    SIMULATE,
    SYMBOL,
    TIMEFRAME,
)
from monitor.heartbeat import Heartbeat
from monitor.telegram import TelegramAlert
from mt5.executor import Executor
from state_machine import State, StateMachine
from tools.market_structure import structure_reference_text


class TradingCEO:
    def __init__(
        self,
        db,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.db = db
        self.symbol = symbol or SYMBOL
        self.timeframe = timeframe or TIMEFRAME
        self.market = MarketAgent(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )
        self.vision = VisionAgent()
        self.macro = MacroAgent()
        self.research = ResearchAgent(timeframe=self.timeframe)
        self.strategy_advisor = StrategyAdvisor()
        self.trader = TraderAgent()
        self.risk = RiskAgent(db)
        self.executor = Executor(db)
        self.memory = Memory(db)
        self.heartbeat = Heartbeat(db)
        self.telegram = TelegramAlert()
        self.sm = StateMachine()
        self.last_decide_bar_ts = None
        self.last_decide_close = None
        self.wake_high = None
        self.wake_low = None
        self.last_position_state = ()

    def run_once(
        self,
        cycle: int = 0,
        manual: bool = False,
        use_vision: bool = True,
    ) -> dict:
        self.executor.sync_positions()
        self.sm.transition(State.ANALYZING, note=f"cycle {cycle}")
        market = self.market.observe()
        manage_events = self.executor.manage(
            market["close"],
            symbol=self.symbol,
        )
        positions = self.executor.positions_summary()
        account = self.risk.account_snapshot()
        pause_reason = self.risk.hard_pause_reason(account)
        if pause_reason is None and self._auto_trading_disabled():
            pause_reason = "MT5 Algo Trading 未开启，订单被拒绝"
        close_price = float(market["close"])
        atr = float(market["indicators"].get("atr") or max(close_price * 0.002, 1.0))
        df = market.get("df")
        current_high = close_price
        current_low = close_price
        if df is not None and len(df) > 0:
            current_high = float(df["high"].iloc[-1])
            current_low = float(df["low"].iloc[-1])
        if self.wake_high is None:
            self.wake_high = current_high
            self.wake_low = current_low
        else:
            self.wake_high = max(self.wake_high, current_high)
            self.wake_low = min(self.wake_low, current_low)
        position_state = self._position_state(positions)
        wake_move = (
            atr > 0
            and self.wake_high is not None
            and self.wake_low is not None
            and (self.wake_high - self.wake_low)
            >= atr * float(DECISION_WAKE_ATR_MULTIPLE)
        )
        wake_position_change = (
            self.last_decide_bar_ts is not None
            and position_state != self.last_position_state
        )
        range_wake_enabled = MID_BAR_WAKE_MODE in (
            "range",
            "position_range",
        )
        position_wake_enabled = MID_BAR_WAKE_MODE in (
            "position",
            "position_range",
        )
        range_wake = range_wake_enabled and wake_move
        position_wake = position_wake_enabled and wake_position_change
        bar_ts = None
        if df is not None and len(df) > 0:
            bar_ts = str(df["time"].iloc[-1])
        if (
            ENABLE_BAR_CLOSE_DECISIONS
            and not manual
            and not pause_reason
            and not range_wake
            and not position_wake
            and bar_ts is not None
            and self.last_decide_bar_ts == bar_ts
        ):
            daily_review = self._maybe_daily_review()
            self.sm.transition(
                State.WAITING,
                note=f"same bar {bar_ts}; waiting for next closed bar",
            )
            return {
                "cycle": cycle,
                "action": "SKIP",
                "confidence": 0,
                "reason": "同一根已完成K线已决策，等待新K线收盘再分析",
                "account": account,
                "chart": None,
                "indicators": self._indicators(market),
                "agents": {"market": market.get("view", {})},
                "research": None,
                "strategy": None,
                "daily_review": daily_review,
                "events": manage_events,
            }
        daily_review = None
        if pause_reason:
            self.sm.transition(State.WAITING, note="hard risk pause")
            self.heartbeat.tick(
                "WAITING",
                f"{pause_reason}; 已暂停AI/视觉/新闻API调用",
            )
            return {
                "cycle": cycle,
                "action": "WAIT",
                "confidence": 0,
                "reason": f"{pause_reason}；已暂停AI/视觉/新闻API调用",
                "risk_reasons": [pause_reason],
                "account": account,
                "chart": None,
                "indicators": self._indicators(market),
                "agents": {
                    "market": market.get("view", {}),
                    "risk": {
                        "allow": False,
                        "reasons": [pause_reason],
                    },
                },
                "research": None,
                "strategy": None,
                "daily_review": daily_review,
                "events": manage_events,
            }
        daily_review = self._maybe_daily_review()
        self.last_decide_bar_ts = bar_ts
        self.last_decide_close = close_price
        self.wake_high = current_high
        self.wake_low = current_low
        self.last_position_state = position_state
        memory = self.memory.context_block()
        macro = self.macro.observe()
        macro_risk = str(macro.get("event_risk") or "low").lower()
        if ENABLE_MACRO_GATE and macro_risk == "high" and not positions:
            self.sm.transition(State.WAITING, note="macro high risk gate")
            self.heartbeat.tick(
                "WAITING",
                "宏观事件风险高，禁止新开仓；持仓管理继续运行",
            )
            return {
                "cycle": cycle,
                "action": "WAIT",
                "confidence": 0,
                "reason": "宏观事件风险高，代码级禁止新开仓",
                "risk_reasons": ["macro high risk gate"],
                "account": account,
                "chart": None,
                "indicators": self._indicators(market),
                "agents": {
                    "market": market.get("view", {}),
                    "macro": macro,
                },
                "research": None,
                "strategy": None,
                "daily_review": daily_review,
                "events": manage_events,
            }

        self.sm.transition(State.PLANNING)
        vision = self.vision.observe(market) if use_vision else {}
        research = (
            self.research.scan(limit=4)
            if not LITE_AGENTS and cycle > 0 and cycle % RESEARCH_INTERVAL == 0
            else None
        )
        strategy = self.strategy_advisor.advise(market, self.db)
        decision = self.trader.decide(
            market,
            vision,
            macro,
            memory,
            strategy,
            positions=positions,
        )
        macro_gate = (
            ENABLE_MACRO_GATE
            and macro_risk == "high"
            and decision["action"] in ("BUY", "SELL")
        )
        if macro_gate:
            gated_action = str(decision["action"])
            decision = {
                **decision,
                "action": "WAIT",
                "confidence": min(float(decision.get("confidence") or 0), 40),
                "reason": (
                    "宏观事件风险高，代码级禁止新开/加仓；"
                    "若持仓出现反向确认仍可执行平仓"
                ),
                "add_on": False,
            }
            decision["strategy_events"] = (decision.get("strategy_events") or []) + [
                {
                    "event_type": "macro_gate",
                    "action": gated_action,
                    "close": market["close"],
                    "rsi": market["indicators"].get("rsi"),
                    "atr": market["indicators"].get("atr"),
                    "reason": decision["reason"],
                }
            ]
        decision["macro_gate"] = macro_gate

        self.sm.transition(State.RISK_CHECK)
        review = self.risk.review(decision, market, positions=positions)
        agents = {
            "market": market.get("view", {}),
            "vision": vision.get("vision", {}),
            "macro": macro,
            "trader": decision,
            "risk": {"allow": review["allow"], "reasons": review["reasons"]},
        }
        volume = (
            review["order"].get("volume") if review["order"] else None
        )
        self.db.save_decision(
            market["symbol"],
            market["timeframe"],
            decision,
            state="planned",
            cycle=cycle,
            volume=volume,
        )
        for event in decision.get("strategy_events") or []:
            self.db.save_strategy_event(
                {
                    **event,
                    "symbol": market["symbol"],
                    "timeframe": market["timeframe"],
                    "outcome": "blocked" if not review["allow"] else "executed",
                }
            )

        if not review["allow"]:
            self.sm.transition(State.WAITING, note="risk blocked")
            self.heartbeat.tick("WAITING", "; ".join(review["reasons"]))
            return {
                "cycle": cycle,
                "action": "WAIT",
                "confidence": decision["confidence"],
                "reason": decision.get("reason") or "no trade requested",
                "risk_reasons": review["reasons"],
                "account": review["account"],
                "chart": vision.get("chart"),
                "indicators": self._indicators(market),
                "agents": agents,
                "research": research,
                "strategy": strategy,
                "daily_review": daily_review,
            }

        order = review["order"]
        if order:
            order["market_state"] = strategy.get("market_state")
            order["persona"] = strategy.get("preferred_persona")
        if manual:
            try:
                if order["action"] == "CLOSE":
                    prompt = (
                        f"\nManual confirm CLOSE all {order['symbol']} "
                        f"@ {order['entry']}? [y/N] "
                    )
                else:
                    prompt = (
                        f"\nManual confirm {order['action']} {order['symbol']} "
                        f"{order['volume']} lots @ {order['entry']}? [y/N] "
                    )
                confirmed = input(prompt).strip().lower()
            except EOFError:
                confirmed = "n"
            if confirmed != "y":
                self.sm.transition(State.WAITING, note="manual reject")
                self.heartbeat.tick("WAITING", "manual reject")
                return {
                    "cycle": cycle,
                    "action": "REJECTED",
                    "confidence": decision["confidence"],
                    "reason": "manual confirm declined",
                    "account": review["account"],
                    "chart": vision.get("chart"),
                    "indicators": self._indicators(market),
                    "agents": agents,
                    "research": research,
                    "strategy": strategy,
                    "daily_review": daily_review,
                }

        self.sm.transition(State.EXECUTING)
        try:
            trade = self.executor.execute(order)
            self.last_position_state = self._position_state(
                self.executor.positions_summary()
            )
        except Exception as exc:
            reason = f"order failed: {exc}"
            self.sm.transition(State.WAITING, note="order failed")
            self.heartbeat.tick("WAITING", reason)
            self.telegram.send(reason)
            return {
                "cycle": cycle,
                "action": "WAIT",
                "confidence": decision["confidence"],
                "reason": reason,
                "risk_reasons": [reason],
                "account": review["account"],
                "chart": vision.get("chart"),
                "indicators": self._indicators(market),
                "agents": agents,
                "research": research,
                "strategy": strategy,
                "daily_review": daily_review,
                "events": manage_events,
            }
        self.sm.transition(State.MONITORING)
        events = manage_events
        if order["action"] == "CLOSE" and trade and trade.get("events"):
            events = trade["events"]
        if order["action"] == "CLOSE":
            self.db.save_strategy_event(
                {
                    "event_type": "ai_close",
                    "symbol": market["symbol"],
                    "timeframe": market["timeframe"],
                    "action": "CLOSE",
                    "close": market["close"],
                    "rsi": market["indicators"].get("rsi"),
                    "atr": market["indicators"].get("atr"),
                    "box_low": market["indicators"].get("donchian_low"),
                    "box_high": market["indicators"].get("donchian_high"),
                    "two_b_bottom": market["indicators"].get("two_b_bottom"),
                    "two_b_top": market["indicators"].get("two_b_top"),
                    "reason": order.get("reason", ""),
                    "outcome": "executed",
                }
            )

        if order["action"] == "CLOSE":
            message = (
                f"AI CLOSE {order['symbol']} entry={order['entry']} "
                f"closed={trade.get('closed', 0) if trade else 0} "
                f"conf={order['confidence']}"
            )
        else:
            message = (
                f"{order['action']} {order['symbol']} entry={order['entry']} "
                f"sl={order['sl']} tp={order['tp']} vol={order['volume']} "
                f"conf={order['confidence']}"
            )
        self.telegram.send(message)
        self.heartbeat.tick("MONITORING", message)
        self.sm.transition(State.WAITING, note="cycle done")

        return {
            "cycle": cycle,
            "action": order["action"],
            "confidence": order["confidence"],
            "reason": order["reason"],
            "order": order,
            "trade_id": trade.get("id") if trade else None,
            "events": events,
            "account": review["account"],
            "chart": vision.get("chart"),
            "indicators": self._indicators(market),
            "agents": agents,
            "research": research,
            "strategy": strategy,
            "daily_review": daily_review,
        }

    def run_review(self) -> dict:
        self.executor.sync_positions()
        trades = self.db.trades_current_session()
        mt5_summary = self._mt5_summary()
        technical, macro = self._review_context()
        strategy_events = self.db.strategy_events_today()
        strategy_events_text = summarize_strategy_events(strategy_events)
        result = ReflectionAgent().review(
            trades,
            extra=mt5_summary,
            technical=technical,
            macro=macro,
            strategy_events=strategy_events,
            decisions=self.db.decisions_today(),
        )
        self.db.save_review(
            result.get("summary", ""),
            result.get("lesson", ""),
            len(trades),
        )
        self.memory.save_experience(result.get("lesson", ""), "lesson")
        review_dir = os.path.join(DATA_DIR, "reviews")
        os.makedirs(review_dir, exist_ok=True)
        review_path = os.path.join(
            review_dir, datetime.now().date().isoformat() + ".md"
        )
        with open(review_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"## {datetime.now().isoformat(timespec='seconds')}\n\n"
                f"MT5摘要：{mt5_summary or '无'}\n\n"
                f"技术面：{technical or '无'}\n\n"
                f"消息面：{macro or '无'}\n\n"
                f"策略事件：{strategy_events_text or '无'}\n\n"
                f"今日交易{len(trades)}笔\n\n"
                f"摘要：{result.get('summary', '')}\n\n"
                f"经验：{result.get('lesson', '')}\n\n"
            )
        self.heartbeat.tick("REVIEW", "daily review complete")
        return result

    def _review_context(self) -> tuple[str | None, str | None]:
        technical = None
        macro_text = None
        try:
            market = self.market.observe()
            ind = market["indicators"]
            closes = list(ind.get("closes") or [])
            recent = closes[-24:]
            technical = (
                f"最近收盘={ind.get('close')}，趋势={ind.get('trend')}，"
                f"状态={ind.get('market_state')}，RSI={ind.get('rsi')}，"
                f"ATR={ind.get('atr')}，ADX={ind.get('adx')}，"
                f"动量确认={ind.get('momentum_ok')}，震荡信号={ind.get('range_signal')}，"
                f"近24根K线最高={max(recent) if recent else '无'}，"
                f"最低={min(recent) if recent else '无'}，K线数={len(recent)}"
            )
            structure_ref = structure_reference_text(ind.get("structure"))
            if structure_ref:
                technical += f"；市场结构={structure_ref}"
            momentum_ref = ind.get("momentum_reference") or "暂无动量背离"
            key_ref = ind.get("key_level_reference") or "暂无关键位"
            technical += f"；背离={momentum_ref}；关键位={key_ref}"
            technical += (
                f"；CCI={ind.get('cci_reference') or '暂无CCI背离'}"
                f"；ICT={ind.get('ict_reference') or '暂无ICT结构'}"
                f"；MACD={ind.get('macd_reference') or '暂无MACD状态'}"
                f"；阶段={ind.get('trend_phase_reference') or '暂无阶段'}"
            )
            view = market.get("view", {})
            if view.get("summary"):
                technical += f"；市场AI：{view['summary']}"
            df = market.get("df")
            if df is not None and len(df) > 0:
                candle_rows = df.tail(12)
                ohlc = "；".join(
                    (
                        f"{str(row['time'])[5:16]}:"
                        f"O{float(row['open']):.2f}/H{float(row['high']):.2f}/"
                        f"L{float(row['low']):.2f}/C{float(row['close']):.2f}"
                    )
                    for _, row in candle_rows.iterrows()
                )
                technical += f"；最近12根K线OHLC：{ohlc}"
                try:
                    from vision.analyzer import VisionAnalyzer

                    local_vision = VisionAnalyzer._local_analysis(
                        {
                            "trend": ind.get("trend", "neutral"),
                            "bars": [
                                {
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                }
                                for _, row in candle_rows.iterrows()
                            ],
                        }
                    )
                    technical += (
                        f"；K线结构={local_vision.get('structure', '无')}"
                        f"，形态={local_vision.get('setup', '无')}，"
                        f"置信度={local_vision.get('confidence', '无')}"
                    )
                except Exception:
                    technical += "；K线结构分析不可用"
        except Exception as exc:
            technical = f"技术面获取失败: {type(exc).__name__}"
        try:
            item = self.macro.observe()
            headline_text = "；".join(
                f"{h.get('text', '')[:120]}"
                for h in (item.get("headlines") or [])[:4]
            )
            macro_text = (
                f"事件风险={item.get('event_risk')}，"
                f"建议={item.get('recommendation')}，"
                f"来源={item.get('news_source') or 'none'}，"
                f"说明={item.get('summary') or '无'}"
            )
            if headline_text:
                macro_text += f"；最新消息：{headline_text}"
        except Exception as exc:
            macro_text = f"消息面获取失败: {type(exc).__name__}"
        return technical, macro_text

    def run_pre_open(self) -> dict:
        """Wake-up routine: daily review, market scan and a decision briefing."""
        self.executor.sync_positions()
        daily_review = None
        if self.db.reviews_today() == 0:
            try:
                daily_review = self.run_review()
            except Exception as exc:
                daily_review = {
                    "summary": f"复盘失败: {type(exc).__name__}",
                    "lesson": "",
                }
        market = self.market.observe()
        macro = self.macro.observe()
        research = self.research.scan(limit=4)
        briefing = self._pre_open_briefing(market, macro, research)
        self.memory.save_experience(briefing, "preopen")
        return {
            "daily_review": daily_review,
            "market": market,
            "macro": macro,
            "research": research,
            "briefing": briefing,
        }

    @staticmethod
    def _pre_open_briefing(market: dict, macro: dict, research: list[dict]) -> str:
        ind = market["indicators"]
        trend_zh = {"bullish": "看涨", "bearish": "看跌", "neutral": "震荡"}
        state_zh = {"trend": "趋势", "range": "震荡"}
        rsi = float(ind.get("rsi", 50))
        if rsi >= 70:
            sentiment = "偏多过热"
        elif rsi <= 30:
            sentiment = "偏空超卖"
        elif ind.get("trend") == "bullish" and rsi > 55:
            sentiment = "偏多"
        elif ind.get("trend") == "bearish" and rsi < 45:
            sentiment = "偏空"
        else:
            sentiment = "中性"
        technical = (
            f"XAUUSD 状态={state_zh.get(ind.get('market_state', 'range'), '震荡')}，"
            f"趋势={trend_zh.get(ind.get('trend', 'neutral'), '震荡')}，"
            f"RSI={ind.get('rsi')}，ATR={ind.get('atr')}，ADX={ind.get('adx')}，"
            f"动量确认={'是' if ind.get('momentum_ok') else '否'}，"
            f"震荡信号={ind.get('range_signal', 'none')}"
        )
        technical += (
            f"，市场结构={structure_reference_text(ind.get('structure'))}"
        )
        technical += (
            f"，背离={ind.get('momentum_reference') or '暂无动量背离'}"
            f"，关键位={ind.get('key_level_reference') or '暂无关键位'}"
        )
        technical += (
            f"，CCI={ind.get('cci_reference') or '暂无CCI背离'}"
            f"，ICT={ind.get('ict_reference') or '暂无ICT结构'}"
            f"，MACD={ind.get('macd_reference') or '暂无MACD状态'}"
            f"，阶段={ind.get('trend_phase_reference') or '暂无阶段'}"
        )
        macro_text = (
            f"事件风险={macro.get('event_risk', '低')}，"
            f"建议={macro.get('recommendation', '正常')}"
        )
        scan_text = "；".join(
            f"{item.get('symbol')} score={item.get('score')} "
            f"{str(item.get('summary', ''))[:50]}"
            for item in research
        )
        return (
            f"开市前简报。技术面：{technical}。"
            f"市场情绪：{sentiment}。消息面：{macro_text}。"
            f"机会扫描：{scan_text or '无'}。"
            "首轮决策请综合技术面、市场情绪和消息面，按风控规则执行。"
        )

    def _mt5_summary(self) -> str | None:
        if not self.executor.real_execution:
            return None
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, timedelta

            positions = mt5.positions_get() or []
            my_positions = [p for p in positions if p.magic == MT5_MAGIC]
            floating = sum(float(p.profit or 0.0) for p in my_positions)
            now_utc = datetime.utcnow()
            since_utc = now_utc - timedelta(days=1)
            deals = mt5.history_deals_get(since_utc, now_utc) or []
            my_deals = [d for d in deals if d.magic == MT5_MAGIC]
            out_entries = {getattr(mt5, "DEAL_ENTRY_OUT", 1), getattr(mt5, "DEAL_ENTRY_INOUT", 0)}
            out_deals = [d for d in my_deals if d.entry in out_entries]
            closed_positions = {d.position_id for d in out_deals}
            realized = sum(float(d.profit or 0.0) for d in out_deals)
            return (
                f"MT5持仓{len(my_positions)}笔，浮动盈亏{floating:.2f} USD；"
                f"近24小时平仓{len(closed_positions)}笔，已实现盈亏{realized:.2f} USD"
            )
        except Exception as exc:
            return f"MT5摘要获取失败: {type(exc).__name__}"

    def status(self) -> dict:
        return {
            "state": self.sm.state.value,
            "transitions": self.sm.summary(),
        }

    def _maybe_daily_review(self) -> dict | None:
        """Run the nightly review once per day after DAILY_REVIEW_HOUR."""
        if datetime.now().hour < DAILY_REVIEW_HOUR:
            return None
        if self.db.reviews_today() > 0:
            return None
        return self.run_review()

    @staticmethod
    def _indicators(market: dict) -> dict:
        return {
            key: value
            for key, value in market["indicators"].items()
            if key != "closes"
        }

    @staticmethod
    def _auto_trading_disabled() -> bool:
        """True only for real MT5 execution when Algo Trading is turned off."""
        if SIMULATE or EXECUTION_MODE != "mt5":
            return False
        try:
            import MetaTrader5 as mt5

            terminal = mt5.terminal_info()
            return terminal is not None and not bool(terminal.trade_allowed)
        except Exception:
            return False

    @staticmethod
    def _position_state(positions: list | None) -> tuple:
        """Stable key for detecting position changes between cycles."""
        return tuple(
            sorted(
                (
                    str(pos.get("side", "")).upper(),
                    str(pos.get("ticket") or ""),
                    str(pos.get("symbol") or ""),
                )
                for pos in (positions or [])
            )
        )
