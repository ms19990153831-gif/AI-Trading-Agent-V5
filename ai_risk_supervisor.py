"""AI risk supervisor: external drawdown/macro gate for the deterministic EA.

The deterministic EA keeps trading. This process only:
- tracks account equity peak and closes EA positions on drawdown threshold
- asks the macro/news agent for event risk
- writes normal/reduce/halt into MQL5/Files/ai_risk.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5

from agent.macro_agent import MacroAgent


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def close_ea_positions(magic: int, symbol: str, reason: str) -> int:
    positions = mt5.positions_get(symbol=symbol) or []
    closed = 0
    for pos in positions:
        if pos.magic != magic:
            continue
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue
        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos.volume),
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 30,
            "magic": magic,
            "comment": f"risk:{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            log(f"close failed ticket={pos.ticket} err={mt5.last_error()}")
            continue
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log(f"close rejected ticket={pos.ticket} retcode={result.retcode}")
            continue
        closed += 1
        log(f"closed ticket={pos.ticket} volume={pos.volume} reason={reason}")
    return closed


def decide_level(macro: dict, dd_pct: float, close_dd: float) -> tuple[str, str]:
    if dd_pct >= close_dd:
        return "halt", f"equity drawdown {dd_pct:.2f} >= {close_dd:.1f}%"
    event_risk = str(macro.get("event_risk") or "low").lower()
    recommendation = str(macro.get("recommendation") or "normal").lower()
    if recommendation == "stand_down" or event_risk == "high":
        return "halt", f"macro {event_risk} / {recommendation}"
    if recommendation == "reduce_position":
        return "reduce", f"macro reduce: {str(macro.get('summary') or '')[:120]}"
    return "normal", f"macro {event_risk} / normal"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--magic", type=int, default=888889)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--close-dd", type=float, default=15.0)
    parser.add_argument("--halt-cooldown-minutes", type=int, default=60)
    args = parser.parse_args()

    if not mt5.initialize(r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        log(f"MT5 init failed: {mt5.last_error()}")
        return 1
    account = mt5.account_info()
    if account is None:
        log("No MT5 account available")
        mt5.shutdown()
        return 1
    data_path = Path(getattr(mt5.terminal_info(), "data_path", ""))
    risk_file = data_path / "MQL5" / "Files" / "ai_risk.txt"
    state_file = Path(__file__).resolve().parent / "data" / "ai_risk_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    risk_file.parent.mkdir(parents=True, exist_ok=True)

    log(f"AI risk supervisor running magic={args.magic} close_dd={args.close_dd}%")
    macro = MacroAgent()
    peak = account.equity
    halt_until = 0.0

    while True:
        try:
            account = mt5.account_info()
            if account is None:
                log("Account lost, waiting")
                time.sleep(args.interval)
                continue
            equity = float(account.equity)
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak * 100.0 if peak > 0 else 0.0

            macro_view = macro.observe()
            level, reason = decide_level(macro_view, dd_pct, args.close_dd)
            now = time.time()
            if level == "halt":
                halt_until = now + args.halt_cooldown_minutes * 60
            elif now < halt_until:
                level = "halt"
                reason = f"halt cooldown remains {int((halt_until - now) / 60)}m"

            if level == "halt":
                close_ea_positions(args.magic, args.symbol, "halt")

            state = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "level": level,
                "reason": reason,
                "event_risk": macro_view.get("event_risk"),
                "macro_recommendation": macro_view.get("recommendation"),
                "equity": round(equity, 2),
                "peak": round(peak, 2),
                "drawdown_pct": round(dd_pct, 2),
            }
            state_file.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            risk_file.write_text(
                f"{level}|risk supervisor|{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}|{dd_pct:.2f}",
                encoding="ascii",
                errors="ignore",
            )
            log(
                f"state={level} reason={reason} equity={equity:.2f} "
                f"peak={peak:.2f} dd={dd_pct:.2f}%"
            )
        except KeyboardInterrupt:
            log("AI risk supervisor stopped by user")
            break
        except Exception as exc:
            log(f"cycle error: {type(exc).__name__}: {exc}")
        time.sleep(args.interval)

    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
