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
| R8a | Stops must be anchored to ATR/structural distance, not ultra-tight | code | implemented |
| R9 | Trend state + momentum_ok => BUY/SELL by trend direction | prompt+code | implemented |
| R10 | Range state + range_signal => follow signal | prompt | implemented |
| R11 | Allow WAIT only with explicit reasons | prompt | implemented |
| R12 | Existing position blocks new entries; add-on limited to one per symbol | code | implemented |
| R13 | Range box boundary structure can be an entry | prompt | implemented |
| R14 | Vision stop/stall candles affect new entries and exits | prompt+code | partial |
| R15 | BOS/CHoCH/trend health is context | advisory | implemented |
| R16 | Momentum divergence and key levels are context | advisory | implemented |
| R17 | CCI/MACD/ICT are context | advisory | implemented |
| R18 | Trend phase guides whether to chase or wait | advisory/prompt | implemented |
| R19 | Structure vs SMA conflict must wait for confirmation | code | implemented |
| R20 | Rule veto cannot be argued away | code | implemented |
| R21 | Macro risk is advisory, not a hard stop | advisory/prompt | implemented |

## Versioning

- **V1 implemented**: the rule table above is the running rule set.
- **V2 draft**: proposed changes that still need calendar data or more evidence:
  - Macro event calendar with automatic confidence increase and position halving.
  - CHoCH should not immediately discard the old trend until structure confirmation.
  - Severe momentum divergence should raise entry quality, not automatically forbid a trade.
  - Optional counter-SMA entry only at a large-timeframe supply/demand zone after CHoCH.

## Pipeline protocol

Decision requests pass through the guards in this order. Once a guard returns
`REJECT`, later signal matching does not force the order through:

```text
Signal generation (R9/R10/R13)
  -> Vision/form filters (R14)
  -> Structure conflict guard (R19)
  -> Macro/event risk context (R21)
  -> Rule veto (R20)
  -> Risk firewall (R4/R5/R12)
  -> ATR/structural stop guard (R8a)
  -> Order build and lot sizing
```

## Soft rules vs hard rules

| Layer | Owner | Examples |
| --- | --- | --- |
| Soft | LLM prompt | market interpretation, visual structure, confidence score, `reason` text |
| Advisory | LLM context | divergence, ICT, CHoCH, macro risk, key levels |
| Hard | Python code | confidence floor, risk %, daily loss, max positions, add-on limit, breakeven requirement, ATR stop guard, min-lot risk |

## How to add a rule

1. Give it a unique id, for example `R22`.
2. Describe the exact market state, signal, filters, and stop/exit behavior.
3. Decide whether it should be prompt advice or hard code.
4. Add unit tests and a backtest result before opening a PR.
