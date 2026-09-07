# AI Trading Agent V1-V5

An AI-driven MT5 trading agent. It is not a fixed EA: it combines market data,
rule-based guards, LLM decisions, vision analysis, macro/news context, risk
firewalls, paper/live execution, daily reviews, and an extensible rule library.

> This project can submit real orders when live trading is enabled. Start with
> a demo account, use restricted permissions, and never rely on it as
> investment advice.

## Public repository note

This repository contains source, tests, and deployment files only. It does not
commit `.env`, market CSV files, SQLite databases, logs, or generated charts.

Setup:

```bash
cp .env.example .env
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Feature stages

| Mode | Purpose |
| --- | --- |
| V1 | AI + MT5 data + manual confirmation loop |
| V2 | Automated execution with order manager, paper broker, SL/TP, breakeven, trailing |
| V3 | Chart generation and vision analysis |
| V4 | Daily review and lesson extraction |
| V5 | Full multi-agent 24h loop with heartbeat and Telegram alerts |

Run a quick offline simulation:

```bash
python main.py --mode demo --cycles 3
```

## Real environment

Copy `.env.example` to `.env` and fill in your LLM key and MT5 details:

```dotenv
SIMULATE=false
USE_MOCK_LLM=false
EXECUTION_MODE=paper # or mt5
DEEPSEEK_KEY=sk-xxx
MT5_LOGIN=0
```

Run the production loop:

```bash
python main.py --mode v5 --cycles 288 --interval 300
```

## Multi-symbol / multi-timeframe universe

`TRADING_UNIVERSE` accepts `SYMBOL:TIMEFRAME` entries separated by semicolons:

```dotenv
TRADING_UNIVERSE=XAUUSD:H1;XAUUSD:M15;EURUSD:H1;GBPUSD:H1
```

Each combination has its own agent state. The same symbol can only hold one
direction at a time; different symbols share `MAX_OPEN_POSITIONS`.

## Trading rules

- Human-readable rule library: [docs/RULES.md](docs/RULES.md)
- Machine-readable manifest: [rules/manifest.json](rules/manifest.json)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)

Rules live in three layers:

| Layer | Meaning |
| --- | --- |
| prompt | Guidance sent to the LLM |
| code | Deterministic logic that cannot be bypassed |
| advisory | Context only, never a standalone trigger |

## Risk model

- Per-trade risk is controlled by `MAX_RISK_PERCENT`
- Daily loss limit is `DAILY_LOSS_LIMIT`
- Max concurrent positions is `MAX_OPEN_POSITIONS`
- Partial take profit at 0.5R
- Breakeven after partial
- Trailing stop after 0.8R
- Target profit at 2R by default

## Disclaimer

This is a research and automation framework. Backtest results are historical
simulations, not guarantees of future performance. Validate on a demo account
before any live deployment.
