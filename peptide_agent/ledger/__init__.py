"""Evidence Ledger — append-only, hash-addressable, replay-grade.

Every claim and decision in PeptideAgent must produce an EvidenceCard which
is committed to the ledger. The Critic uses the ledger as the source of
truth for evidence-gated checks.
"""

from .store import EvidenceLedger

__all__ = ["EvidenceLedger"]
