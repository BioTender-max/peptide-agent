"""Append-only Evidence Ledger.

- JSONL on disk, one card per line.
- Each card gets a deterministic content hash on commit.
- Read API: by id, by source, by content_hash, by tag, by claim_substring.
- Append API: commits one EvidenceCard, computes hash, returns the committed copy.
- Supersession: if `supersedes` is set on the new card, prior cards remain in the
  ledger but `get_active()` filters them out.

Concurrency note: the ledger uses a single writer model (one agent at a time
during a LangGraph step). Multi-writer would need a real DB; that's out of
scope for v0.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from ..schemas import EvidenceCard


def _hash_card(card: EvidenceCard) -> str:
    """Deterministic content hash over the substantive fields.

    Excludes card_id, timestamp, content_hash. Includes claim, source_id,
    source_type, payload (JSON-canonical), extracted_by, derived_from.
    """
    payload = {
        "claim": card.claim,
        "source_id": card.source_id,
        "source_type": card.source_type,
        "extracted_by": card.extracted_by,
        "payload": card.payload,
        "derived_from": sorted(card.derived_from),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class EvidenceLedger:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._index: dict[str, EvidenceCard] = {}
        self._load()

    def _load(self) -> None:
        if self.path.stat().st_size == 0:
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                card = EvidenceCard.model_validate(d)
                self._index[card.card_id] = card

    # ----- Write -----

    def append(self, card: EvidenceCard) -> EvidenceCard:
        if card.content_hash is None:
            card.content_hash = _hash_card(card)
        # Dedupe by content_hash within this run
        for existing in self._index.values():
            if existing.content_hash == card.content_hash and existing.card_id != card.card_id:
                return existing
        self._index[card.card_id] = card
        with self.path.open("a") as f:
            f.write(card.model_dump_json() + "\n")
        return card

    def extend(self, cards: Iterable[EvidenceCard]) -> list[EvidenceCard]:
        return [self.append(c) for c in cards]

    # ----- Read -----

    def get(self, card_id: str) -> Optional[EvidenceCard]:
        return self._index.get(card_id)

    def by_source(self, source_id: str) -> list[EvidenceCard]:
        """Match cards by exact source_id (e.g. a UniProt accession or PDB id)."""
        return [c for c in self._index.values() if c.source_id == source_id]

    def by_source_type(self, source_type: str) -> list[EvidenceCard]:
        """Match cards by source kind: 'uniprot', 'pdb', 'literature', 'tool_output', 'agent_decision'."""
        return [c for c in self._index.values() if c.source_type == source_type]

    def by_tag(self, tag: str) -> list[EvidenceCard]:
        return [c for c in self._index.values() if c.tag == tag]

    def search_claim(self, substring: str) -> list[EvidenceCard]:
        s = substring.lower()
        return [c for c in self._index.values() if s in c.claim.lower()]

    def get_active(self) -> list[EvidenceCard]:
        """Cards not superseded by any later card."""
        superseded = {c.supersedes for c in self._index.values() if c.supersedes}
        return [c for c in self._index.values() if c.card_id not in superseded]

    def all(self) -> list[EvidenceCard]:
        return list(self._index.values())

    def __len__(self) -> int:
        return len(self._index)
