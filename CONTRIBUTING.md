# Contributing to PeptideAgent

Thanks for taking a look. This is research code and PRs are welcome, but a few
ground rules keep the agent system honest.

## Before sending a PR

```bash
# Lint
ruff check .

# Type check (currently advisory; failures don't block)
mypy peptide_agent/

# Tests
pytest tests/
```

All six smoke tests must pass on Python 3.10 / 3.11 / 3.12.

## Style

- Type hints on every public function. `Optional`/`|` syntax is fine.
- Pydantic models for every agent-visible artifact. No bare dicts crossing agent boundaries.
- Use `Annotated` reducers in `state.py` whenever a field can be appended-to by parallel branches.

## Adding a new tool wrapper

Every external call must go through `peptide_agent/tools/`. The contract:

1. Pure function or thin class — **no LangGraph state inside**.
2. Returns plain dicts or Pydantic models — never partial agents.
3. Network calls have a 15–30 s timeout.
4. Cache to `cache/<tool>_<key>.json` when the same input is likely to repeat.

When an agent uses a new tool, it MUST emit a corresponding `EvidenceCard`
into the ledger so the Critic can audit it.

## Adding an evidence axis

If you add a new signal (e.g. literature mutagenesis residues, ConSurf scores,
SASA), update three places:

1. The tool: deterministic function in `peptide_agent/tools/`.
2. The agent that uses it: emit an `EvidenceCard` with `source_type` set.
3. `docs/status.md`: bump the "axis count" and re-check `Hotspot.role` Literal.

## Don't fabricate data

This project went through a class of bug where the assistant baked in
"canonical hotspot" lists from memory rather than from retrieved literature.
The fix was to delete the set, not to dress it up. Same rule applies here:

- If a residue, citation, or threshold comes from prior knowledge instead of
  a tool output, **either compute it or drop it**. Don't ship it.
- Every quantitative claim in the README, in a docstring, or in a paper-style
  output must trace back to a real ledger entry.

## Reporting bugs

Include:
- Python version + `pip list` output (or `uv pip list`).
- A minimal reproducer (a `state = {...}` and the agent call that fails).
- The full traceback.

Bonus: the `cache/` and `runs/` artifacts produced by the failing run.
