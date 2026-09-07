---
name: Rule suggestion
about: Propose a new trading rule or an update to an existing rule
title: "[RULE] R22 - short description"
labels: rule
assignees: ''
---

## Proposed rule

Give the rule one stable id, for example `R22`.

## Market state

Which condition should this rule apply to? Trend, range, reversal, news, or another state?

## Trigger

What exactly needs to be true before the rule acts?

## Expected behavior

What should the AI do when the rule fires? Open, wait, close, reduce risk, or add-on?

## Evidence

Describe the scenario or backtest result that supports this rule.

## Layer

- [ ] Prompt advice
- [ ] Hard code
- [ ] Context/advisory only

## Checklist

- [ ] Added the rule to `docs/RULES.md`
- [ ] Added the rule to `rules/manifest.json`
- [ ] Added a unit test
- [ ] No secrets, keys, logs, databases, or market CSV files are included
