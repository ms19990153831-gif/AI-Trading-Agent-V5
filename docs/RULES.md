# Trading Rules Library

This file is the canonical human-readable rule library. The machine-readable
copy lives in `rules/manifest.json`.

Every rule has one stable id. A rule may live in one of three layers:

| Layer | Meaning |
| --- | --- |
| `prompt` | Guidance sent to the LLM in `brain/prompt.py` |
| `code` | Deterministic logic that cannot be bypassed by the LLM |
| `advisory` | Context/reference only, never a standalone trigger |

## Rule table

| ID | Rule | Layer | Status |
| --- | --- | --- | --- |
| R1 | No signal edge => WAIT | prompt | implemented |
| R2 | Strict JSON decision output | code/prompt | implemented |
| R3 | Do not trade against the major trend without reversal structure | prompt | implemented |
| R4 | Risk before prediction | code | implemented |
| R5 | Confidence floor 70; R9/R10 can use 60-75 | code | implemented |
| R6 | Chinese reason output | prompt | implemented |
| R7 | Strong trend + momentum can allow chasing with tight stop | prompt | implemented |
| R8 | Do not wait mechanically on a single RSI value | prompt | implemented |
| R9 | Trend state + momentum_ok => BUY/SELL by trend direction | prompt+code | implemented |
| R10 | Range state + range_signal => follow signal | prompt | implemented |
| R11 | Allow WAIT only with explicit reasons | prompt | implemented |
| R12 | Existing position blocks new entries unless strong add-on | code | implemented |
| R13 | Range box boundary structure can be an entry | prompt | implemented |
| R14 | Vision stop/stall candles affect new entries and exits | prompt+code | partial |
| R15 | BOS/CHoCH/trend health is context | advisory | implemented |
| R16 | Momentum divergence and key levels are context | advisory | implemented |
| R17 | CCI/MACD/ICT are context | advisory | implemented |
| R18 | Trend phase guides whether to chase or wait | advisory/prompt | implemented |
| R19 | Structure vs SMA conflict must wait for confirmation | code | implemented |
| R20 | Rule veto cannot be argued away | code | implemented |
| R21 | Macro risk is advisory, not a hard stop | advisory/prompt | implemented |

## How to add a rule

1. Give it a unique id, for example `R22`.
2. Describe the exact market state, signal, filters, and stop/exit behavior.
3. Decide whether it should be prompt advice or hard code.
4. Add unit tests and a backtest result before opening a PR.
