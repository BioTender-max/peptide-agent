"""PeptideAgent-v0: agentic peptide binder discovery system.

A 7-agent LangGraph system that runs:
    target brief -> deep research -> structure/epitope analysis ->
    multi-strategy binder design -> in silico evaluation ->
    Critic gating (4 layers) -> reporter.

Every artifact is provenance-tracked through an append-only Evidence Ledger.
"""

__version__ = "0.1.0"
