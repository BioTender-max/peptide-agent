---
name: Bug report
about: Something is broken or behaves unexpectedly
title: '[BUG] '
labels: bug
assignees: ''
---

## Summary
<!-- One sentence: what fails? -->

## Reproducer

```python
# Minimal state + agent call
from peptide_agent.agents import <agent>
state = {...}
<agent>.run(state)
```

## Expected vs actual
- **Expected:**
- **Actual:**

## Environment

- Python version (`python --version`):
- OS:
- `pip list` (or `uv pip list`) — paste the output, or at least the
  versions of `langgraph`, `langchain-core`, `pydantic`, `biotite`,
  `boltz-api`.

## Traceback / smoke-suite output

```
# Paste the full traceback here.
# If the failure is in the smoke suite, paste `python tests/test_smoke_all.py` output.
```

## Bonus

- [ ] `cache/` artefacts produced by the failing run (or a hint of what
      was cached).
- [ ] `runs/` directory state for the failing run.
