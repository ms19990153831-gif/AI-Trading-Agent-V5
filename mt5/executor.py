"""Execution gateway: MT5 real orders or the offline paper broker."""

from __future__ import annotations

from config import (
    BREAKEVEN_AT_RR,
    EXECUTION_MODE,
    MT5_MAGIC,
    PARTIAL_TP_RATIO,
    SYMBOL,
    TRAILING_ACTIVATE_RR,
    TRAILING_DISTANCE_RR,
)
from execution.paper_broker import PaperBroker
from tools.calculator import price_precision


class Executor:
    def __init__(self, db) -> None:
        self.db = db
        self.paper = PaperBroker(db)
        self.real_execution = EXECUTION_MODE == "mt5"

    def execute(self, order: dict) -> dict | None:
        if order["action"] == "WAIT":
            return None
        if order["action"] == "CLOSE":
            if not self.real_execution:
                events = self.paper.close_all(float(order["entry"]))
                return {
                    "action": "CLOSE",
                    "symbol": order.get("symbol", SYMBOL),
                    "entry": order["entry"],
                    "closed": len(events),
                    "events": events,
                }
            return self._mt5_close(order)
        if not self.real_execution:
            return self.paper.place(order)
        return self._mt5_send(order)

    def tick(self, price: float) -> list[dict]:
        if self.real_execution:
            return []
        return self.paper.tick(price)

    def manage(
        self,
        price: float,
        symbol: str | None = None,
    ) -> list[dict]:
        """Manage open positions each cycle: partial TP, breakeven, trailing."""
        if not self.real_execution:
            return self.paper.tick(price)
        return self._mt5_manage(price, symbol=symbol)

    def _mt5_manage(
        self,
        price: float,
        symbol: str | None = None,
    ) -> list[dict]:
        import MetaTrader5 as mt5

        events: list[dict] = []
        positions = mt5.positions_get() or []
        open_trades = {
            int(trade["ticket"]): trade
            for trade in self.db.open_trades()
            if trade.get("ticket")
        }
        for pos in positions:
            if pos.magic != MT5_MAGIC:
                continue
            if symbol and str(pos.symbol).upper() != str(symbol).upper():
                continue
            trade = open_trades.get(int(pos.ticket))
            side = (
                "BUY"
                if pos.type == mt5.POSITION_TYPE_BUY
                else "SELL"
            )
            entry = float(pos.price_open)
            sl = float(pos.sl or 0)
            tp = float(pos.tp or 0)
            original_sl = (
                float(trade["sl"]) if trade and trade.get("sl") else sl
            )
            risk = abs(entry - original_sl)
            if risk <= 0:
                risk = abs(entry - tp) if tp else 0
            if risk <= 0:
                continue
            volume = float(pos.volume)
            original_volume = (
                float(trade["volume"])
                if trade and trade.get("volume")
                else volume
            )
            partial_done = bool(trade and trade.get("partial_done"))
            symbol = pos.symbol

            partial_price = (
                entry + risk * float(BREAKEVEN_AT_RR)
                if side == "BUY"
                else entry - risk * float(BREAKEVEN_AT_RR)
            )
            partial_leaves_minimum = (
                round(
                    original_volume
                    - round(original_volume * float(PARTIAL_TP_RATIO), 2),
                    2,
                )
                >= 0.01
            )
            if (
                not partial_done
                and partial_leaves_minimum
                and (
                    (side == "BUY" and price >= partial_price)
                    or (side == "SELL" and price <= partial_price)
                )
            ):
                close_volume = round(
                    original_volume * float(PARTIAL_TP_RATIO), 2
                )
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    deal_price = tick.bid if side == "BUY" else tick.ask
                    close_type = (
                        mt5.ORDER_TYPE_SELL
                        if side == "BUY"
                        else mt5.ORDER_TYPE_BUY
                    )
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": close_volume,
                        "type": close_type,
                        "position": pos.ticket,
                        "price": deal_price,
                        "deviation": 20,
                        "magic": MT5_MAGIC,
                        "comment": "AI Partial TP",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    if (
                        result is not None
                        and result.retcode == mt5.TRADE_RETCODE_DONE
                    ):
                        if trade:
                            remaining = round(
                                max(original_volume - close_volume, 0.01),
                                2,
                            )
                            self.db.update_trade(
                                trade["id"],
                                partial_done=1,
                                volume=remaining,
                            )
                        events.append(
                            {
                                "type": "PARTIAL_TP",
                                "trade_id": int(pos.ticket),
                                "price": round(deal_price, 2),
                                "volume": close_volume,
                                "pnl": None,
                            }
                        )

            breakeven_price = (
                entry + risk * float(BREAKEVEN_AT_RR)
                if side == "BUY"
                else entry - risk * float(BREAKEVEN_AT_RR)
            )
            if (
                (side == "BUY" and price >= breakeven_price and sl < entry)
                or (side == "SELL" and price <= breakeven_price and sl > entry)
            ):
                if self._modify_sl(symbol, pos.ticket, entry, tp):
                    if trade:
                        self.db.update_trade(trade["id"], sl_current=entry)
                    events.append(
                        {
                            "type": "BREAKEVEN",
                            "trade_id": int(pos.ticket),
                            "price": round(price, 2),
                            "sl": round(entry, 2),
                        }
                    )
                    sl = entry

            activate_price = (
                entry + risk * float(TRAILING_ACTIVATE_RR)
                if side == "BUY"
                else entry - risk * float(TRAILING_ACTIVATE_RR)
            )
            trail_distance = risk * float(TRAILING_DISTANCE_RR)
            new_sl = (
                price - trail_distance
                if side == "BUY"
                else price + trail_distance
            )
            if (
                (side == "BUY" and price >= activate_price and new_sl > sl)
                or (side == "SELL" and price <= activate_price and new_sl < sl)
            ):
                if self._modify_sl(symbol, pos.ticket, new_sl, tp):
                    if trade:
                        self.db.update_trade(
                            trade["id"],
                            sl_current=round(new_sl, 2),
                        )
                    events.append(
                        {
                            "type": "TRAILING",
                            "trade_id": int(pos.ticket),
                            "price": round(price, 2),
                            "sl": round(new_sl, 2),
                        }
                    )
        return events

    def _modify_sl(self, symbol: str, ticket: int, sl: float, tp: float) -> bool:
        import MetaTrader5 as mt5

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": MT5_MAGIC,
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def _mt5_send(self, order: dict) -> dict:
        import MetaTrader5 as mt5

        side = (
            mt5.ORDER_TYPE_BUY if order["action"] == "BUY" else mt5.ORDER_TYPE_SELL
        )
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.get("symbol", SYMBOL),
            "volume": order["volume"],
            "type": side,
            "price": order["entry"],
            "sl": order["sl"],
            "tp": order["tp"],
            "deviation": 20,
            "magic": MT5_MAGIC,
            "comment": "AI Trading Agent",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"MT5 order failed: {result}")
        trade = self.paper.place(
            {
                **order,
                "action": order["action"],
                "entry": result.price,
            }
        )
        self.db.update_trade(trade["id"], ticket=result.order)
        return {
            "id": result.order,
            "side": order["action"],
            "volume": order["volume"],
            "entry": result.price,
            "sl": order["sl"],
            "tp": order["tp"],
            "db_id": trade["id"],
            "ticket": result.order,
        }

    def _mt5_close(self, order: dict) -> dict:
        import MetaTrader5 as mt5

        positions = mt5.positions_get() or []
        closed = 0
        for pos in positions:
            if pos.magic != MT5_MAGIC:
                continue
            if order.get("symbol") and pos.symbol != order["symbol"]:
                continue
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue
            close_type = (
                mt5.ORDER_TYPE_BUY
                if pos.type == mt5.POSITION_TYPE_SELL
                else mt5.ORDER_TYPE_SELL
            )
            price = tick.bid if close_type == mt5.ORDER_TYPE_BUY else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": float(pos.volume),
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": MT5_MAGIC,
                "comment": "AI Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
        self.sync_positions()
        return {
            "action": "CLOSE",
            "symbol": order.get("symbol", SYMBOL),
            "entry": order.get("entry"),
            "closed": closed,
            "events": [],
        }

    def sync_positions(self) -> dict:
        """Sync MT5 closed positions back into the local trade database."""
        if not self.real_execution:
            return {"closed": 0}
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta

        open_trades = self.db.open_trades()
        since = datetime.now() - timedelta(days=7)
        # Broker/terminal clocks can run ahead of the local machine, so give
        # the query an extra day instead of ending exactly at "now".
        until = datetime.now() + timedelta(days=1)
        deals = mt5.history_deals_get(since, until) or []
        positions = mt5.positions_get() or []
        open_tickets = {p.ticket for p in positions if p.magic == MT5_MAGIC}
        position_by_ticket = {
            int(p.ticket): p for p in positions if p.magic == MT5_MAGIC
        }

        closed_count = 0
        for trade in open_trades:
            ticket = trade.get("ticket")
            if not ticket:
                # Legacy orders recorded before tickets existed: fuzzy match.
                deal_type = (
                    mt5.DEAL_TYPE_BUY if trade["side"] == "BUY" else mt5.DEAL_TYPE_SELL
                )
                entry = float(trade["entry"])
                volume = float(trade["volume"])
                for deal in deals:
                    if deal.magic != MT5_MAGIC or deal.symbol != trade["symbol"]:
                        continue
                    if deal.type != deal_type or deal.entry != mt5.DEAL_ENTRY_IN:
                        continue
                    if abs(float(deal.volume) - volume) > 0.001:
                        continue
                    if abs(float(deal.price) - entry) > 6.0:
                        continue
                    ticket = deal.position_id
                    self.db.update_trade(trade["id"], ticket=ticket)
                    break
            if ticket and ticket in position_by_ticket:
                current = position_by_ticket[ticket]
                updates = {}
                actual_volume = round(float(current.volume), 2)
                if abs(actual_volume - float(trade["volume"] or 0.0)) > 0.001:
                    updates["volume"] = actual_volume
                digits = price_precision(trade.get("symbol"))
                actual_sl = (
                    round(float(current.sl), digits) if current.sl else None
                )
                old_sl = (
                    round(float(trade.get("sl_current") or 0.0), digits)
                    if trade.get("sl_current")
                    else None
                )
                if actual_sl is not None and actual_sl != old_sl:
                    updates["sl_current"] = actual_sl
                if updates:
                    self.db.update_trade(trade["id"], **updates)
            if not ticket or ticket in open_tickets:
                continue
            out_deals = [
                deal
                for deal in deals
                if getattr(deal, "position_id", None) == ticket
                and deal.entry == mt5.DEAL_ENTRY_OUT
            ]
            if not out_deals:
                continue
            pnl = sum(float(deal.profit or 0.0) for deal in out_deals)
            exit_ts = datetime.utcfromtimestamp(
                max(deal.time for deal in out_deals)
            ).isoformat() + "Z"
            self.db.update_trade(
                trade["id"],
                status="closed",
                exit_price=round(
                    float(out_deals[-1].price or 0.0),
                    price_precision(trade.get("symbol")),
                ),
                exit_ts=exit_ts,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / max(float(trade["volume"]) * float(trade["entry"]) * 100, 1.0) * 100, 4),
            )
            closed_count += 1
        return {"closed": closed_count}

    def positions_summary(self) -> list[dict]:
        """Current AI positions used as decision context and add-on gating."""
        if self.real_execution:
            try:
                import MetaTrader5 as mt5

                positions = mt5.positions_get() or []
                return [
                    {
                        "side": (
                            "BUY"
                            if p.type == mt5.POSITION_TYPE_BUY
                            else "SELL"
                        ),
                        "volume": float(p.volume),
                        "entry": float(p.price_open),
                        "profit": float(p.profit or 0.0),
                        "symbol": p.symbol,
                        "ticket": int(p.ticket),
                    }
                    for p in positions
                    if p.magic == MT5_MAGIC
                ]
            except Exception:
                return []
        return [
            {
                "side": trade["side"],
                "volume": float(trade["volume"]),
                "entry": float(trade["entry"]),
                "profit": 0.0,
                "symbol": trade.get("symbol"),
                "ticket": trade.get("ticket"),
            }
            for trade in self.db.open_trades()
        ]
