"""AI Trading Agent - V1 through V5 entry point.

Modes:
  v1    AI + MT5 data + manual confirmation loop
  v2    V1 + automatic paper execution with order manager and exits
  v3    V2 + candlestick chart generation and vision analysis
  v4    Daily reflection and lesson extraction
  v5    Full multi-agent 24h production loop
  demo  V5 with a bounded offline simulation and summary
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from datetime import datetime

from agent.ceo import TradingCEO
from config import (
    ALLOW_WEEKEND,
    DB_PATH,
    HEARTBEAT_FILE,
    MARKET_HOURS_ENABLED,
    MID_BAR_WAKE_MODE,
    PAPER_STEP_SECONDS,
    PRE_OPEN_MINUTES,
    SIMULATE,
    SYMBOL,
    TIMEFRAME,
    TRADER_PID_FILE,
    TRADING_UNIVERSE,
)
from market_hours import market_is_open, next_open, preopen_time
from memory.database import Database
from mt5.connector import MT5Connector
from backtest import run_backtest
from training import run_training


_ZH_TREND = {"bullish": "看涨", "bearish": "看跌", "neutral": "震荡"}
_ZH_RISK = {"low": "低", "medium": "中", "high": "高"}
_ZH_STRUCTURE = {
    "Higher High Higher Low": "高点更高 低点更高",
    "Lower High Lower Low": "低点更低 高点更低",
    "Pullback within uptrend": "上涨趋势中的回调",
    "Rebound within downtrend": "下跌趋势中的反弹",
    "range": "震荡",
}
_ZH_SETUP = {
    "Potential long": "潜在做多",
    "Potential short": "潜在做空",
    "No setup": "无形态",
}
_ZH_MACRO = {
    "normal": "正常",
    "reduce_position": "减仓",
    "stand_down": "停止交易",
}


def zh(value, mapping):
    return mapping.get(str(value), str(value))


def zh_risk(text):
    out = re.sub(r"confidence (\d+) below (\d+)", r"置信度 \1 低于 \2", str(text))
    out = re.sub(r"risk ([\d.]+%) above ([\d.]+%)", r"风险 \1 高于 \2", out)
    return (
        out.replace("no trade requested", "未请求交易")
        .replace("max open positions reached", "已达最大持仓数")
        .replace("daily loss limit reached", "已达当日亏损上限")
        .replace("loss streak paused the trader", "连亏暂停交易")
    )


def run_loop(
    ceo: TradingCEO,
    cycles: int,
    manual: bool,
    use_vision: bool,
    interval: float,
    market_hours: bool = False,
) -> list[dict]:
    results: list[dict] = []
    for cycle in range(1, cycles + 1):
        if market_hours:
            now = datetime.now()
            if not market_is_open(now):
                wake = preopen_time(now)
                if wake > now:
                    print(
                        f"市场休市，AI暂停。将于开市前{PRE_OPEN_MINUTES}分钟唤醒: "
                        f"{wake.strftime('%Y-%m-%d %H:%M')}"
                    )
                    time.sleep(max(1.0, (wake - now).total_seconds()))
                pre = ceo.run_pre_open()
                if pre.get("daily_review"):
                    print(f"  [每日复盘] {pre['daily_review'].get('summary', '')}")
                    print(f"               {pre['daily_review'].get('lesson', '')}")
                print("  [开市前扫描]")
                for finding in pre.get("research", []):
                    print(
                        f"    {finding.get('symbol')}: score={finding.get('score')} "
                        f"{str(finding.get('summary', ''))[:80]}"
                    )
                print(f"  [开市前简报] {pre.get('briefing', '')}")
                now2 = datetime.now()
                if not market_is_open(now2):
                    open_dt = next_open(now2)
                    print(f"等待开盘: {open_dt.strftime('%Y-%m-%d %H:%M')}")
                    time.sleep(max(1.0, (open_dt - now2).total_seconds()))
                continue
        try:
            result = ceo.run_once(
                cycle=cycle,
                manual=manual,
                use_vision=use_vision,
            )
        except Exception as exc:
            ceo.sm.reset()
            traceback.print_exc()
            print(f"[cycle {cycle}] cycle error: {exc}; continuing after pause")
            result = {
                "cycle": cycle,
                "action": "WAIT",
                "confidence": 0,
                "reason": f"cycle error: {type(exc).__name__}: {exc}",
                "risk_reasons": ["cycle error, AI restarted from WAITING"],
                "account": None,
                "chart": None,
                "indicators": {},
                "agents": {},
                "research": None,
                "strategy": None,
                "daily_review": None,
                "events": [],
            }
        if result.get("action") == "SKIP":
            indicators = result.get("indicators", {})
            account = result.get("account") or {}
            print(
                f"[cycle {cycle}] 维护: close={indicators.get('close')} "
                f"positions={len(account.get('positions') or [])} "
                f"同H1已决策，等待新K线/盘中唤醒={MID_BAR_WAKE_MODE}"
            )
            for event in result.get("events", []):
                print(
                    f"[skip] {event['type']} price={event.get('price')} "
                    f"pnl={event.get('pnl')}"
                )
            daily_review = result.get("daily_review")
            if daily_review:
                print(f"[每日复盘] {daily_review.get('summary')}")
                print(f"           {daily_review.get('lesson')}")
            if interval > 0:
                time.sleep(interval)
            continue
        results.append(result)
        indicators = result.get("indicators", {})
        print(f"\n[cycle {cycle}] {result['action']:<8} conf={result['confidence']}")
        if indicators:
            print(
                f"  市场: close={indicators.get('close')} "
                f"trend={zh(indicators.get('trend'), _ZH_TREND)} "
                f"rsi={indicators.get('rsi')} "
                f"atr={indicators.get('atr')}"
            )
        agents = result.get("agents", {})
        market_view = agents.get("market", {})
        if market_view:
            print(
                f"  [市场AI] trend={zh(market_view.get('trend'), _ZH_TREND)} "
                f"summary={str(market_view.get('summary', ''))[:90]}"
            )
        vision_view = agents.get("vision", {})
        if vision_view:
            print(
                f"  [视觉AI] trend={zh(vision_view.get('trend'), _ZH_TREND)} "
                f"structure={zh(vision_view.get('structure'), _ZH_STRUCTURE)} "
                f"setup={zh(vision_view.get('setup'), _ZH_SETUP)} "
                f"confidence={vision_view.get('confidence')} "
                f"source={vision_view.get('source', '?')}"
            )
        macro_view = agents.get("macro", {})
        if macro_view:
            print(
                f"  [宏观AI] event_risk={zh(macro_view.get('event_risk'), _ZH_RISK)} "
                f"recommendation={zh(macro_view.get('recommendation'), _ZH_MACRO)} "
                f"source={macro_view.get('news_source', '?')} "
                f"headlines={len(macro_view.get('headlines') or [])}"
            )
        research = result.get("research")
        if research:
            print("  [研究AI] 机会扫描:")
            for finding in research:
                print(
                    f"    {finding.get('symbol')}: score={finding.get('score')} "
                    f"trend={finding.get('trend')} "
                    f"{str(finding.get('summary', ''))[:70]}"
                )
        strategy = result.get("strategy")
        if strategy:
            print(
                f"  [策略顾问] 状态={strategy.get('market_state')} "
                f"建议={strategy.get('preferred_persona')}（仅供参考，AI自主决策）"
            )
        daily_review = result.get("daily_review")
        if daily_review:
            print(f"  [每日复盘] {daily_review.get('summary')}")
            print(f"             {daily_review.get('lesson')}")
        if result.get("reason"):
            print(f"  AI策略: {result['reason']}")
        if result.get("risk_reasons"):
            print(f"  风控: {zh_risk(', '.join(result['risk_reasons']))}")
        if result.get("order"):
            order = result["order"]
            print(
                f"  订单方案: {order['action']} {order['symbol']} "
                f"vol={order.get('volume', 'CLOSE')} entry={order.get('entry')} "
                f"sl={order.get('sl')} tp={order.get('tp')}"
            )
        if result.get("chart"):
            print(f"           chart: {result['chart']}")
        for event in result.get("events", []):
            print(f"           event: {event['type']} price={event.get('price')} "
                  f"pnl={event.get('pnl')} vol={event.get('volume')}")
        if interval > 0:
            time.sleep(interval)
    return results


def build_trading_ceos(db: Database) -> list[TradingCEO]:
    universe = TRADING_UNIVERSE or [(SYMBOL, TIMEFRAME)]
    return [
        TradingCEO(db, symbol=symbol, timeframe=timeframe)
        for symbol, timeframe in universe
    ]


def run_multi_loop(
    ceos: list[TradingCEO],
    cycles: int,
    manual: bool,
    use_vision: bool,
    interval: float,
    market_hours: bool = False,
) -> list[dict]:
    """Run one trading universe: every configured symbol/timeframe is
    evaluated once per cycle while sharing the same risk and MT5 account."""
    results: list[dict] = []
    for cycle in range(1, cycles + 1):
        if market_hours:
            now = datetime.now()
            if not market_is_open(now):
                wake = preopen_time(now)
                if wake > now:
                    print(
                        f"市场休市，AI暂停。将于开市前{PRE_OPEN_MINUTES}分钟唤醒: "
                        f"{wake.strftime('%Y-%m-%d %H:%M')}"
                    )
                    time.sleep(max(1.0, (wake - now).total_seconds()))
                pre = ceos[0].run_pre_open()
                if pre.get("daily_review"):
                    print(f"  [每日复盘] {pre['daily_review'].get('summary', '')}")
                    print(f"               {pre['daily_review'].get('lesson', '')}")
                print("  [开市前扫描]")
                for finding in pre.get("research", []):
                    print(
                        f"    {finding.get('symbol')}: score={finding.get('score')} "
                        f"{str(finding.get('summary', ''))[:80]}"
                    )
                print(f"  [开市前简报] {pre.get('briefing', '')}")
                now2 = datetime.now()
                if not market_is_open(now2):
                    open_dt = next_open(now2)
                    print(f"等待开盘: {open_dt.strftime('%Y-%m-%d %H:%M')}")
                    time.sleep(max(1.0, (open_dt - now2).total_seconds()))
                continue
        for ceo in ceos:
            label = f"{ceo.symbol} {ceo.timeframe}"
            try:
                result = ceo.run_once(
                    cycle=cycle,
                    manual=manual,
                    use_vision=use_vision,
                )
            except Exception as exc:
                ceo.sm.reset()
                traceback.print_exc()
                print(
                    f"[{label} cycle {cycle}] cycle error: {exc}; "
                    "continuing after pause"
                )
                result = {
                    "cycle": cycle,
                    "action": "WAIT",
                    "confidence": 0,
                    "reason": f"cycle error: {type(exc).__name__}: {exc}",
                    "risk_reasons": ["cycle error, AI restarted from WAITING"],
                    "account": None,
                    "chart": None,
                    "indicators": {},
                    "agents": {},
                    "research": None,
                    "strategy": None,
                    "daily_review": None,
                    "events": [],
                }
            if result.get("action") == "SKIP":
                indicators = result.get("indicators", {})
                account = result.get("account") or {}
                print(
                    f"[{label} cycle {cycle}] 维护: "
                    f"close={indicators.get('close')} "
                    f"positions={len(account.get('positions') or [])} "
                    f"同周期已决策，等待新K线/盘中唤醒={MID_BAR_WAKE_MODE}"
                )
                for event in result.get("events", []):
                    print(
                        f"[skip] {event['type']} price={event.get('price')} "
                        f"pnl={event.get('pnl')}"
                    )
                daily_review = result.get("daily_review")
                if daily_review:
                    print(f"[每日复盘] {daily_review.get('summary')}")
                    print(f"           {daily_review.get('lesson')}")
                continue
            results.append(result)
            indicators = result.get("indicators", {})
            print(
                f"\n[{label} cycle {cycle}] "
                f"{result['action']:<8} conf={result['confidence']}"
            )
            if indicators:
                print(
                    f"  市场: close={indicators.get('close')} "
                    f"trend={zh(indicators.get('trend'), _ZH_TREND)} "
                    f"rsi={indicators.get('rsi')} "
                    f"atr={indicators.get('atr')}"
                )
            agents = result.get("agents", {})
            market_view = agents.get("market", {})
            if market_view:
                print(
                    f"  [市场AI] trend={zh(market_view.get('trend'), _ZH_TREND)} "
                    f"summary={str(market_view.get('summary', ''))[:90]}"
                )
            vision_view = agents.get("vision", {})
            if vision_view:
                print(
                    f"  [视觉AI] trend={zh(vision_view.get('trend'), _ZH_TREND)} "
                    f"structure={zh(vision_view.get('structure'), _ZH_STRUCTURE)} "
                    f"setup={zh(vision_view.get('setup'), _ZH_SETUP)} "
                    f"confidence={vision_view.get('confidence')} "
                    f"source={vision_view.get('source', '?')}"
                )
            macro_view = agents.get("macro", {})
            if macro_view:
                print(
                    f"  [宏观AI] event_risk={zh(macro_view.get('event_risk'), _ZH_RISK)} "
                    f"recommendation={zh(macro_view.get('recommendation'), _ZH_MACRO)} "
                    f"source={macro_view.get('news_source', '?')}"
                )
            daily_review = result.get("daily_review")
            if daily_review:
                print(f"  [每日复盘] {daily_review.get('summary')}")
                print(f"             {daily_review.get('lesson')}")
            if result.get("reason"):
                print(f"  AI策略: {result['reason']}")
            if result.get("risk_reasons"):
                print(
                    f"  风控: {zh_risk(', '.join(result['risk_reasons']))}"
                )
            if result.get("order"):
                order = result["order"]
                print(
                    f"  订单方案: {order['action']} {order['symbol']} "
                    f"vol={order.get('volume', 'CLOSE')} "
                    f"entry={order.get('entry')} sl={order.get('sl')} "
                    f"tp={order.get('tp')}"
                )
            if result.get("chart"):
                print(f"           chart: {result['chart']}")
            for event in result.get("events", []):
                print(
                    f"           event: {event['type']} "
                    f"price={event.get('price')} pnl={event.get('pnl')} "
                    f"vol={event.get('volume')}"
                )
        if interval > 0:
            time.sleep(interval)
    return results


def build_summary(db: Database, results: list[dict]) -> dict:
    closed = db.closed_trades()
    open_positions = db.open_trades()
    experiences = db.get_recent_experiences(3)
    chart_paths = [r["chart"] for r in results if r.get("chart")]
    return {
        "simulate": SIMULATE,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "cycles": len(results),
        "open_positions": len(open_positions),
        "closed_trades": len(closed),
        "net_pnl": round(sum(float(t["pnl"] or 0.0) for t in closed), 2),
        "charts": chart_paths,
        "latest_lessons": [e["content"] for e in experiences],
    }


def run_analyze(ceo: TradingCEO, db: Database) -> dict:
    """Run every agent once against current market data; no orders are sent."""
    market = ceo.market.observe()
    vision = ceo.vision.observe(market)
    macro = ceo.macro.observe()
    memory = ceo.memory.context_block()
    decision = ceo.trader.decide(market, vision, macro, memory)
    review = ceo.risk.review(decision, market)
    db.save_decision(
        market["symbol"],
        market["timeframe"],
        decision,
        state="analyze",
        cycle=0,
    )
    indicators = dict(market["indicators"])
    indicators.pop("closes", None)
    return {
        "data_source": "MT5" if not SIMULATE else "simulation",
        "market": indicators,
        "chart": vision.get("chart"),
        "vision": vision.get("vision"),
        "macro": macro,
        "decision": decision,
        "risk": {
            "allow": review["allow"],
            "reasons": review["reasons"],
        },
        "order": review["order"],
        "account": review["account"],
        "memory_lessons": [row["content"] for row in memory],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trading Agent V1-V5")
    parser.add_argument(
        "--mode",
        choices=["v1", "v2", "v3", "v4", "v5", "demo", "analyze", "train", "backtest"],
        default="demo",
        help="feature stage to run (default: demo)",
    )
    parser.add_argument("--cycles", type=int, default=None, help="number of cycles")
    parser.add_argument("--manual", action="store_true", help="confirm every order")
    parser.add_argument("--no-vision", action="store_true", help="disable vision layer")
    parser.add_argument("--interval", type=float, default=0.0, help="seconds between cycles")
    parser.add_argument("--skip-review", action="store_true", help="skip nightly review in v5")
    parser.add_argument("--force-review", action="store_true", help="run review even if one already exists today")
    parser.add_argument("--fresh", action="store_true", help="reset local trade database before running")
    parser.add_argument("--environments", type=int, default=20, help="training environments")
    parser.add_argument("--bars", type=int, default=2500, help="history bars for training")
    parser.add_argument("--real-llm", action="store_true", help="use real LLM in backtest")
    parser.add_argument("--start", default="", help="backtest start date YYYY-MM-DD")
    parser.add_argument("--end", default="", help="backtest end date YYYY-MM-DD")
    parser.add_argument("--step", type=int, default=1, help="backtest sampling step (every Nth bar)")
    parser.add_argument("--no-market-hours", action="store_true", help="disable market-hours gating")
    parser.add_argument(
        "--start-at",
        default="",
        help="wait until HH:MM before starting the live loop (e.g. 05:45)",
    )
    args = parser.parse_args()
    if args.mode in ("v1", "v2", "v3", "v4", "v5", "demo"):
        try:
            with open(TRADER_PID_FILE, "w", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
        except OSError:
            pass
    market_hours_enabled = MARKET_HOURS_ENABLED and not args.no_market_hours

    if args.fresh:
        for path in (DB_PATH, HEARTBEAT_FILE):
            try:
                os.remove(path)
            except OSError:
                pass

    if (
        args.mode in ("v1", "v2", "v3", "v5")
        and not ALLOW_WEEKEND
        and datetime.now().weekday() >= 5
        and not market_hours_enabled
    ):
        print(
            "今天是周末，交易循环默认关闭（ALLOW_WEEKEND=false）。"
            "如需强制运行，请在 .env 设置 ALLOW_WEEKEND=true。"
        )
        return

    if args.start_at:
        try:
            hour, minute = (int(part) for part in args.start_at.split(":"))
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            print("--start-at 格式应为 HH:MM，例如 05:45")
            return
        if target > now:
            seconds = (target - now).total_seconds()
            print(f"当前尚未到开盘时间，等待到 {target.isoformat(timespec='minutes')} 再启动...")
            time.sleep(seconds)

    if (
        args.mode in ("v1", "v2", "v3", "v5")
        and not ALLOW_WEEKEND
        and datetime.now().weekday() >= 5
        and not market_hours_enabled
    ):
        print(
            "当前是周末，交易循环默认关闭（ALLOW_WEEKEND=false）。"
            "如需强制运行，请在 .env 设置 ALLOW_WEEKEND=true。"
        )
        return

    db = Database()
    connector = MT5Connector()
    if not SIMULATE:
        connector.connect()
        if connector.connected:
            try:
                import MetaTrader5 as mt5

                info = mt5.account_info()
                terminal = mt5.terminal_info()
                if info is not None:
                    mode = (
                        "DEMO"
                        if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
                        else "REAL"
                    )
                    print(
                        f"MT5 connected: login={info.login} server={info.server} "
                        f"mode={mode} trade_allowed={bool(info.trade_allowed)}"
                    )
                if terminal is not None and not bool(terminal.trade_allowed):
                    print(
                        "警告: MT5 Algo Trading 未开启，订单会被拒绝。"
                        "请在 MT5 工具栏点击 Algo Trading 按钮后重启 AI。"
                    )
            except Exception as exc:
                print(f"MT5 account check failed: {type(exc).__name__}: {exc}")

    ceos = build_trading_ceos(db)
    ceo = ceos[0]
    results: list[dict] = []
    use_vision = not args.no_vision

    def run_review_if_due(force: bool = False) -> dict | None:
        if ceo.db.reviews_today() > 0 and not force:
            print(
                f"[每日复盘] 今天已有 {ceo.db.reviews_today()} 条复盘，"
                "跳过（--force-review 可强制重跑）"
            )
            return None
        review = ceo.run_review()
        print("REVIEW summary:", review.get("summary"))
        print("REVIEW lesson :", review.get("lesson"))
        return review

    if args.mode == "v1":
        results = run_loop(
            ceo,
            args.cycles or 1,
            manual=True,
            use_vision=False,
            interval=args.interval,
            market_hours=market_hours_enabled,
        )
    elif args.mode == "v2":
        results = run_loop(
            ceo,
            args.cycles or 5,
            manual=args.manual,
            use_vision=False,
            interval=args.interval,
            market_hours=market_hours_enabled,
        )
    elif args.mode == "v3":
        results = run_loop(
            ceo,
            args.cycles or 3,
            manual=args.manual,
            use_vision=True,
            interval=args.interval,
            market_hours=market_hours_enabled,
        )
    elif args.mode == "v5":
        results = run_multi_loop(
            ceos,
            args.cycles or 288,
            manual=args.manual,
            use_vision=use_vision,
            interval=args.interval,
            market_hours=market_hours_enabled,
        )
        if not args.skip_review:
            run_review_if_due(force=args.force_review)
    elif args.mode == "v4":
        run_review_if_due(force=args.force_review)
    elif args.mode == "analyze":
        report = run_analyze(ceo, db)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.mode == "train":
        report = run_training(
            db,
            environments=args.environments,
            bars=args.bars,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.mode == "backtest":
        report = run_backtest(
            bars=args.bars,
            use_mock=not args.real_llm,
            start=args.start or None,
            end=args.end or None,
            step=args.step,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        cycles = args.cycles or 5
        results = run_loop(
            ceo,
            cycles,
            manual=args.manual,
            use_vision=use_vision,
            interval=args.interval,
            market_hours=market_hours_enabled and args.mode != "demo",
        )
        if not args.skip_review:
            run_review_if_due(force=args.force_review)

    connector.shutdown()
    if args.mode not in ("v4", "analyze", "train", "backtest"):
        summary = build_summary(db, results)
        print("\n=== SUMMARY ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
