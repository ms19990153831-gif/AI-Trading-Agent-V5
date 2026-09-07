"""Risk agent: applies the firewall and builds a safe order."""

from __future__ import annotations

from risk.manager import RiskManager


class RiskAgent:
    def __init__(self, db) -> None:
        self.manager = RiskManager(db)

    def review(
        self,
        decision: dict,
        market: dict,
        positions: list | None = None,
    ) -> dict:
        return self.manager.review(decision, market, positions=positions)

    def hard_pause_reason(self, account: dict | None = None) -> str | None:
        return self.manager.hard_pause_reason(account)

    def account_snapshot(self) -> dict:
        return self.manager.account.snapshot()
