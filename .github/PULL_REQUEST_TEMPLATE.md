# Pull request

## Summary
<!-- One-paragraph description of what this change does. -->

## Why
<!-- Motivation: bug fix, feature, refactor. Link any related issue. -->

## Validation
<!-- Tick every applicable box. Be honest. -->

- [ ] `ruff check .` passes.
- [ ] `pytest tests/test_smoke.py` passes (6 tests).
- [ ] `python tests/test_smoke_all.py` passes (S1–S7, 0 failures).
- [ ] If this changes a Pydantic schema: I updated every agent that emits
      or consumes the field, and the smoke suite still passes byte-exact
      against the captured oracle.
- [ ] If this changes Prediction scoring weights: I regenerated the T2
      oracle and noted the new values in CHANGELOG.
- [ ] If this adds a new tool wrapper: I added at least one EvidenceCard
      with `source_type` set, and a docs note in `docs/architecture.md`.

## Evidence-ledger discipline (mandatory)

PeptideAgent's audit story depends on every external claim landing in the
ledger. Confirm:

- [ ] No fabricated residues, citations, or thresholds. If a number comes
      from prior knowledge instead of a real tool output, I computed it
      or dropped it.
- [ ] No bare dicts crossing agent boundaries — all artefacts are
      Pydantic models.
- [ ] No raw network calls inside an agent — every external call goes
      through `peptide_agent/tools/`.

## Screenshots / logs
<!-- Paste smoke-suite output or relevant evidence-card excerpts here. -->
