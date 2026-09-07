"""Compare closed PnL between the AI magic and the pure EA magic."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--magics", type=int, nargs="+", default=[888890, 888889])
    args = parser.parse_args()

    if not mt5.initialize(r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        print("MT5 init failed", mt5.last_error())
        return 1
    info = mt5.account_info()
    if info is None:
        print("No account")
        mt5.shutdown()
        return 1

    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    since_dt = since.replace(tzinfo=None)
    now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    deals = mt5.history_deals_get(since_dt, now_dt) or []

    print(f"login={info.login} server={info.server} period={args.hours}h")
    print("magic,trades,wins,losses,gross_win,gross_loss,net,volume")
    for magic in args.magics:
        pnl = 0.0
        wins = 0
        losses = 0
        volume = 0.0
        gross_win = 0.0
        gross_loss = 0.0
        for deal in deals:
            if deal.magic != magic:
                continue
            if deal.entry not in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                continue
            value = float(deal.profit or 0.0) + float(deal.commission or 0.0) + float(deal.swap or 0.0)
            pnl += value
            volume += float(deal.volume or 0.0)
            if value > 0:
                wins += 1
                gross_win += value
            elif value < 0:
                losses += 1
                gross_loss += value
        print(
            f"{magic},{wins + losses},{wins},{losses},"
            f"{gross_win:.2f},{gross_loss:.2f},{pnl:.2f},{volume:.2f}"
        )

    positions = mt5.positions_get() or []
    for magic in args.magics:
        my = [p for p in positions if p.magic == magic]
        floating = sum(float(p.profit or 0.0) for p in my)
        print(f"open magic={magic} count={len(my)} floating={floating:.2f}")

    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
