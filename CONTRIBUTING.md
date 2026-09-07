# Contributing

Thanks for helping us turn the trading rules into a maintainable system.

## Quick start

```bash
git clone https://github.com/ms19990153831-gif/AI-Trading-Agent-V5.git
cd AI-Trading-Agent-V5
copy .env.example .env
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## How to add or improve a rule

1. Open an issue first and describe the rule idea, expected market state, entry,
   stop, and validation method.
2. Update `docs/RULES.md` and `rules/manifest.json` so the rule has one stable id.
3. Add the rule as prompt guidance in `brain/prompt.py`, or as deterministic
   logic in `tools/` / `risk/` when the rule must not be bypassed.
4. Add at least one unit test proving when the rule fires and when it does not.
5. Run `python -m unittest discover -s tests -v`.
6. Submit a pull request.

## Rules of contribution

- Never commit `.env`, API keys, MT5 account details, logs, databases, or market CSV files.
- A rule needs evidence: a test, a scenario, or a backtest result.
- Avoid cherry-picked winning periods. Prefer multi-regime evidence.
- Backtests are research results, not promises of future performance.

## Rule lifecycle

```text
Idea -> Discussion issue -> Draft rule -> Unit test -> Backtest -> Merge
```
