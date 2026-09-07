"""Deterministic position sizing and stop/target math."""

from __future__ import annotations


SYMBOL_SPECS = {
    "XAUUSD": {
        "contract_size": 100,
        "quote_currency": "USD",
        "digits": 2,
        "spread": 0.25,
    },
    "XAGUSD": {
        "contract_size": 5000,
        "quote_currency": "USD",
        "digits": 3,
        "spread": 0.02,
    },
    "EURUSD": {
        "contract_size": 100000,
        "quote_currency": "USD",
        "digits": 5,
        "spread": 0.0002,
    },
    "GBPUSD": {
        "contract_size": 100000,
        "quote_currency": "USD",
        "digits": 5,
        "spread": 0.0002,
    },
    "AUDUSD": {
        "contract_size": 100000,
        "quote_currency": "USD",
        "digits": 5,
        "spread": 0.0002,
    },
    "NZDUSD": {
        "contract_size": 100000,
        "quote_currency": "USD",
        "digits": 5,
        "spread": 0.0002,
    },
    "USDJPY": {
        "contract_size": 100000,
        "quote_currency": "JPY",
        "digits": 3,
        "spread": 0.02,
    },
    "USDCHF": {
        "contract_size": 100000,
        "quote_currency": "CHF",
        "digits": 5,
        "spread": 0.0002,
    },
    "USDCAD": {
        "contract_size": 100000,
        "quote_currency": "CAD",
        "digits": 5,
        "spread": 0.0002,
    },
}


def symbol_spec(symbol: str) -> dict | None:
    """Return contract details for symbols the order sizer understands."""
    value = SYMBOL_SPECS.get(str(symbol or "").upper())
    if value:
        return dict(value)
    return None


def quote_to_usd_factor(symbol: str, price: float) -> float:
    """Convert one quote-currency point into USD for position sizing."""
    spec = symbol_spec(symbol)
    if not spec:
        return 0.0
    currency = str(spec.get("quote_currency") or "USD").upper()
    if currency == "USD":
        return 1.0
    if price > 0:
        return 1.0 / price
    return 0.0


def price_precision(symbol: str) -> int:
    spec = symbol_spec(symbol)
    if not spec:
        return 5
    return int(spec.get("digits") or 2)


def symbol_spread(symbol: str) -> float:
    spec = symbol_spec(symbol)
    if not spec:
        return 0.0
    return float(spec.get("spread") or 0.0)


def risk_money(balance: float, risk_percent: float) -> float:
    return balance * risk_percent


def lot_size(
    balance: float,
    risk_percent: float,
    stop_distance: float,
    contract_size: float = 100,
    quote_to_usd: float = 1.0,
) -> float:
    """Risk-based lot sizing in quote-USD terms."""
    if stop_distance <= 0:
        return 0.01
    factor = max(contract_size * quote_to_usd, 1e-12)
    raw = risk_money(balance, risk_percent) / (stop_distance * factor)
    return max(0.01, round(raw, 2))


def stop_loss_risk_percent(
    balance: float,
    volume: float,
    stop_distance: float,
    contract_size: float = 100,
    quote_to_usd: float = 1.0,
) -> float:
    """Return the stop-loss risk as a fraction of account balance."""
    if min(balance, volume, stop_distance) <= 0:
        return 0.0
    return (volume * stop_distance * contract_size * quote_to_usd) / balance


def default_stop(close: float, atr: float, multiplier: float = 1.5) -> float:
    return max(atr * multiplier, close * 0.001)


def default_target(close: float, stop_distance: float, rr: float) -> float:
    return close + stop_distance * rr


def normalize_risk(risk: float) -> float:
    """Models sometimes express risk as 1 (=1%) instead of 0.01."""
    if risk >= 1.0:
        return round(risk / 100.0, 4)
    return round(risk, 4)
