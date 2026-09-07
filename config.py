"""Central configuration for the AI Trading Agent.

Everything can be overridden through environment variables or a .env file.
The defaults intentionally run a full offline simulation so the project can
be tried without MT5 credentials or an LLM API key.
"""

import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: str) -> bool:
    return _env(key, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: str) -> int:
    try:
        return int(_env(key, default))
    except ValueError:
        return int(default)


def _env_float(key: str, default: str) -> float:
    try:
        return float(_env(key, default))
    except ValueError:
        return float(default)


def _env_list(key: str, default: str) -> list[str]:
    raw = _env(key, default)
    parts = [
        item.strip().upper()
        for item in raw.replace(";", ",").split(",")
        if item.strip()
    ]
    return parts or [default.strip().upper()]


# Market ----------------------------------------------------------------------
SYMBOL = _env("SYMBOL", "XAUUSD")
TIMEFRAME = _env("TIMEFRAME", "H1")
SYMBOLS = _env_list("SYMBOLS", SYMBOL)
TIMEFRAMES = _env_list("TIMEFRAMES", TIMEFRAME)
TRADING_UNIVERSE: list[tuple[str, str]] = []
_raw_universe = _env("TRADING_UNIVERSE", "").strip()
if _raw_universe:
    for _entry in _raw_universe.split(";"):
        _parts = _entry.strip().split(":")
        if len(_parts) == 2 and _parts[0].strip() and _parts[1].strip():
            pair = (_parts[0].strip().upper(), _parts[1].strip().upper())
            if pair not in TRADING_UNIVERSE:
                TRADING_UNIVERSE.append(pair)
if not TRADING_UNIVERSE:
    for _symbol in SYMBOLS:
        for _timeframe in TIMEFRAMES:
            pair = (_symbol, _timeframe)
            if pair not in TRADING_UNIVERSE:
                TRADING_UNIVERSE.append(pair)
BAR_COUNT = _env_int("BAR_COUNT", "200")

# Simulation ------------------------------------------------------------------
SIMULATE = _env_bool("SIMULATE", "true")
USE_MOCK_LLM = _env_bool("USE_MOCK_LLM", "true")
SIM_SEED = _env_int("SIM_SEED", "20260821")
SIM_VOLATILITY = _env_float("SIM_VOLATILITY", "2.5")

# Execution mode: "paper" keeps orders offline, "mt5" sends real MT5 orders.
EXECUTION_MODE = _env("EXECUTION_MODE", "paper").strip().lower()

# LLM -------------------------------------------------------------------------
DEEPSEEK_KEY = _env("DEEPSEEK_KEY", "")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = _env("LLM_MODEL", "deepseek-chat")
VISION_MODEL = _env("VISION_MODEL", "")
VISION_API_KEY = _env("VISION_API_KEY", "") or DEEPSEEK_KEY
VISION_BASE_URL = _env("VISION_BASE_URL", "") or DEEPSEEK_BASE_URL
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", "60")

# MT5 -------------------------------------------------------------------------
MT5_LOGIN = _env_int("MT5_LOGIN", "0")
MT5_PASSWORD = _env("MT5_PASSWORD", "")
MT5_SERVER = _env("MT5_SERVER", "")
MT5_PATH = _env("MT5_PATH", "")
MT5_MAGIC = _env_int("MT5_MAGIC", "888888")

# Account / risk --------------------------------------------------------------
INITIAL_BALANCE = _env_float("INITIAL_BALANCE", "10000")
MAX_RISK_PERCENT = _env_float("MAX_RISK_PERCENT", "0.01")
DAILY_LOSS_LIMIT = _env_float("DAILY_LOSS_LIMIT", "0.03")
MAX_LOSS_STREAK = _env_int("MAX_LOSS_STREAK", "5")
MAX_OPEN_POSITIONS = _env_int("MAX_OPEN_POSITIONS", "3")
ENABLE_MIN_LOT_RISK_GUARD = _env_bool("ENABLE_MIN_LOT_RISK_GUARD", "true")
# Allow small rounding overshoot beyond MAX_RISK_PERCENT; 0.5 means 50%.
MIN_LOT_RISK_TOLERANCE = _env_float("MIN_LOT_RISK_TOLERANCE", "0.5")
ENABLE_DAILY_LOSS_LIMIT = _env_bool("ENABLE_DAILY_LOSS_LIMIT", "true")
ENABLE_LOSS_PAUSE = _env_bool("ENABLE_LOSS_PAUSE", "true")
ENABLE_SIDE_LOSS_PAUSE = _env_bool("ENABLE_SIDE_LOSS_PAUSE", "false")
SIDE_LOSS_PAUSE_STREAK = _env_int("SIDE_LOSS_PAUSE_STREAK", "2")
SIDE_LOSS_PAUSE_HOURS = _env_float("SIDE_LOSS_PAUSE_HOURS", "24")
ENABLE_BAR_CLOSE_DECISIONS = _env_bool("ENABLE_BAR_CLOSE_DECISIONS", "true")
DECISION_WAKE_ATR_MULTIPLE = _env_float("DECISION_WAKE_ATR_MULTIPLE", "0.8")
MID_BAR_WAKE_MODE = _env("MID_BAR_WAKE_MODE", "off")
ENABLE_MACRO_GATE = _env_bool("ENABLE_MACRO_GATE", "false")
ENABLE_TREND_CONFLICT_GUARD = _env_bool("ENABLE_TREND_CONFLICT_GUARD", "true")
ENABLE_RULE_VETO = _env_bool("ENABLE_RULE_VETO", "false")
RULE_VETO_GAP = _env_int("RULE_VETO_GAP", "2")
RULE_VETO_MASK = _env("RULE_VETO_MASK", "base,structure,rsi,cci,macd,ict")
MIN_CONFIDENCE = _env_int("MIN_CONFIDENCE", "70")
TREND_MIN_CONFIDENCE = _env_int("TREND_MIN_CONFIDENCE", "60")
RANGE_MIN_CONFIDENCE = _env_int("RANGE_MIN_CONFIDENCE", "60")
STRUCTURE_CONFIDENCE = _env_int("STRUCTURE_CONFIDENCE", "62")
STRUCTURE_SWING_LEN = _env_int("STRUCTURE_SWING_LEN", "10")
TREND_HEALTH_WEAK_LEVEL = _env_int("TREND_HEALTH_WEAK_LEVEL", "40")
RSI_DIVERGENCE_PIVOT_LEN = _env_int("RSI_DIVERGENCE_PIVOT_LEN", "10")
RSI_DIVERGENCE_MAX_DISTANCE = _env_int("RSI_DIVERGENCE_MAX_DISTANCE", "60")
DIVERGENCE_VALID_BARS = _env_int("DIVERGENCE_VALID_BARS", "20")
CCI_LENGTH = _env_int("CCI_LENGTH", "20")
CCI_DIVERGENCE_EXTREME = _env_float("CCI_DIVERGENCE_EXTREME", "150")
CCI_DIVERGENCE_MAX_DISTANCE = _env_int("CCI_DIVERGENCE_MAX_DISTANCE", "60")
KEY_ZONE_SWING_LEN = _env_int("KEY_ZONE_SWING_LEN", "10")
KEY_ZONE_ATR = _env_float("KEY_ZONE_ATR", "0.75")
ICT_SWEEP_LOOKBACK = _env_int("ICT_SWEEP_LOOKBACK", "20")
ICT_FVG_LOOKBACK = _env_int("ICT_FVG_LOOKBACK", "30")
BOX_BOUNDARY_ATR = _env_float("BOX_BOUNDARY_ATR", "0.6")
ENABLE_STRUCTURE_SIGNAL = _env_bool("ENABLE_STRUCTURE_SIGNAL", "true")
ENABLE_REVERSAL_SIGNAL = _env_bool("ENABLE_REVERSAL_SIGNAL", "true")
ENABLE_BOUNDARY_GUARD = _env_bool("ENABLE_BOUNDARY_GUARD", "true")
ENABLE_VISION_GUARD = _env_bool("ENABLE_VISION_GUARD", "true")
ENABLE_EXTREME_GUARD = _env_bool("ENABLE_EXTREME_GUARD", "false")
ENABLE_TREND_EXHAUSTION_GUARD = _env_bool("ENABLE_TREND_EXHAUSTION_GUARD", "false")
TREND_EXHAUSTION_ADX_MIN = _env_float("TREND_EXHAUSTION_ADX_MIN", "30")
TREND_EXHAUSTION_RSI_SELL_MAX = _env_float("TREND_EXHAUSTION_RSI_SELL_MAX", "30")
TREND_EXHAUSTION_RSI_BUY_MIN = _env_float("TREND_EXHAUSTION_RSI_BUY_MIN", "70")
TREND_EXHAUSTION_EDGE_ATR = _env_float("TREND_EXHAUSTION_EDGE_ATR", "0.8")
EXTREME_RSI_SELL = _env_float("EXTREME_RSI_SELL", "30")
EXTREME_RSI_BUY = _env_float("EXTREME_RSI_BUY", "70")
EXTREME_EDGE_ATR = _env_float("EXTREME_EDGE_ATR", "1.2")
EXTREME_MOVE_ATR = _env_float("EXTREME_MOVE_ATR", "1.5")
BOUNDARY_OVERSOLD_RSI = _env_float("BOUNDARY_OVERSOLD_RSI", "32")
BOUNDARY_OVERBOUGHT_RSI = _env_float("BOUNDARY_OVERBOUGHT_RSI", "68")
REVERSAL_CONFIDENCE = _env_int("REVERSAL_CONFIDENCE", "66")
REVERSAL_CONFIRM_ATR = _env_float("REVERSAL_CONFIRM_ATR", "2.0")
REVERSAL_MAX_MOVE_ATR = _env_float("REVERSAL_MAX_MOVE_ATR", "2.5")
CLOSE_MIN_CONFIDENCE = _env_int("CLOSE_MIN_CONFIDENCE", "60")
CLOSE_RSI_OVERSOLD = _env_float("CLOSE_RSI_OVERSOLD", "45")
CLOSE_RSI_OVERBOUGHT = _env_float("CLOSE_RSI_OVERBOUGHT", "55")
ADDON_MIN_CONFIDENCE = _env_int("ADDON_MIN_CONFIDENCE", "78")
ADDON_MIN_ADX = _env_float("ADDON_MIN_ADX", "28")
ENABLE_QUALITY_GATE = _env_bool("ENABLE_QUALITY_GATE", "true")
ENABLE_HTF_FILTER = _env_bool("ENABLE_HTF_FILTER", "false")
HTF_ADX_MIN = _env_float("HTF_ADX_MIN", "18")
HTF_RSI_BUY_MIN = _env_float("HTF_RSI_BUY_MIN", "45")
HTF_RSI_SELL_MAX = _env_float("HTF_RSI_SELL_MAX", "55")
ENABLE_AI_CLOSE = _env_bool("ENABLE_AI_CLOSE", "true")
ADX_TREND_MIN = _env_float("ADX_TREND_MIN", "24")
RSI_TREND_BUY_MIN = _env_float("RSI_TREND_BUY_MIN", "40")
RSI_TREND_BUY_MAX = _env_float("RSI_TREND_BUY_MAX", "68")
RSI_TREND_SELL_MIN = _env_float("RSI_TREND_SELL_MIN", "32")
RSI_TREND_SELL_MAX = _env_float("RSI_TREND_SELL_MAX", "60")
RANGE_BUY_RSI_MAX = _env_float("RANGE_BUY_RSI_MAX", "55")
RANGE_SELL_RSI_MIN = _env_float("RANGE_SELL_RSI_MIN", "55")
DEFAULT_RR = _env_float("DEFAULT_RR", "2.0")
ATR_STOP_MULTIPLIER = _env_float("ATR_STOP_MULTIPLIER", "1.5")

# V2 order manager ------------------------------------------------------------
PARTIAL_TP_RATIO = _env_float("PARTIAL_TP_RATIO", "0.5")
TRAILING_ACTIVATE_RR = _env_float("TRAILING_ACTIVATE_RR", "0.8")
TRAILING_DISTANCE_RR = _env_float("TRAILING_DISTANCE_RR", "0.6")
BREAKEVEN_AT_RR = _env_float("BREAKEVEN_AT_RR", "0.5")
PAPER_STEP_SECONDS = _env_float("PAPER_STEP_SECONDS", "2")

# Execution costs (backtest realism)
SPREAD = _env_float("SPREAD", "0.25")
COMMISSION_PER_LOT = _env_float("COMMISSION_PER_LOT", "0.0")

# V5 production ---------------------------------------------------------------
HEARTBEAT_INTERVAL = _env_int("HEARTBEAT_INTERVAL", "60")
DAILY_REVIEW_HOUR = _env_int("DAILY_REVIEW_HOUR", "23")
TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID", "")
MACRO_RISK = _env("MACRO_RISK", "low")

# Cost reduction --------------------------------------------------------------
RESEARCH_INTERVAL = _env_int("RESEARCH_INTERVAL", "10")
CONTEXT_CLOSES = _env_int("CONTEXT_CLOSES", "16")
LITE_AGENTS = _env_bool("LITE_AGENTS", "true")

# Live news / macro context ---------------------------------------------------
NEWS_ENABLED = _env_bool("NEWS_ENABLED", "true")
NEWS_FETCH_TIMEOUT = _env_float("NEWS_FETCH_TIMEOUT", "8")
NEWS_MAX_ITEMS = _env_int("NEWS_MAX_ITEMS", "12")
MACRO_CACHE_SECONDS = _env_int("MACRO_CACHE_SECONDS", "300")

# Weekend guard: live trading modes are skipped on Sat/Sun by default.
ALLOW_WEEKEND = _env_bool("ALLOW_WEEKEND", "false")

# Market hours gate for live modes (local time).
MARKET_HOURS_ENABLED = _env_bool("MARKET_HOURS_ENABLED", "true")
MARKET_OPEN_TIME = _env("MARKET_OPEN_TIME", "06:00")
MARKET_CLOSE_TIME = _env("MARKET_CLOSE_TIME", "05:00")
PRE_OPEN_MINUTES = _env_int("PRE_OPEN_MINUTES", "5")

# Paths -----------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
CHART_DIR = os.path.join(DATA_DIR, "charts")
LOG_DIR = os.path.join(DATA_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "trading.db")
HEARTBEAT_FILE = os.path.join(DATA_DIR, "heartbeat.txt")
TRADER_PID_FILE = os.path.join(DATA_DIR, "trader.pid")

for _dir in (DATA_DIR, CHART_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)
