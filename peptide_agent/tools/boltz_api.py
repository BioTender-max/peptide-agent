# v2-RECONSTRUCTED from transcript+spec (worker termination 2026-06-27)
# Verbatim sources: L607 (SDK schema dump), L623 (live validation), L653 (mocked T2)
# Spec source: critic_config.json + step5_smoke_results.json + boltz_api.com/docs/api/
"""Boltz Compute API wrapper.

Routes:
  - estimate_cost  → POST /compute/v1/predictions/structure-and-binding/estimate-cost
  - start          → POST /compute/v1/predictions/structure-and-binding
  - retrieve(id)   → GET  /compute/v1/predictions/structure-and-binding/{id}
  - design.start   → POST /compute/v1/protein/design (binder generation, unused in PD-L1 demo)

The wrapper exposes thin functions the Prediction Agent calls.

# v2 SCHEMA NOTE (verified live 2026-06-27 against boltz-api SDK):
#   - InputEntityBoltz2ProteinEntity uses keys: chain_ids (list[str]), type ("protein"),
#     value (str — the aa sequence), msa (optional). NOT id/sequence/msa-string.
#   - msa is either omitted (→ auto), or {"type": "empty"} for single-sequence mode,
#     or {"type": "custom", "source": {...}} for user-provided MSAs.
#   - Top-level input has keys: entities (Required), binding, num_samples, plus optional
#     bonds/constraints/templates/model_options.
#   - binding for peptide–protein: {"type": "protein_protein_binding",
#                                    "binder_chain_ids": [<peptide_chain>]}.
#   - num_samples = how many independent diffusion seeds the server runs per call.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from boltz_api import Boltz


# ----------------------------------------------------------------------
# Wrapper
# ----------------------------------------------------------------------


class BoltzAPIWrapper:
    """Thin, mockable wrapper around the boltz-api SDK.

    All methods return plain dicts so tests can substitute a MagicMock without
    needing Pydantic stubs. The wrapper does NOT handle polling or cost gating;
    those live in the Prediction Agent which owns the policy.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("BOLTZ_API_KEY")
        if not key:
            raise RuntimeError("BOLTZ_API_KEY not set")
        self.client = Boltz(api_key=key)

    # ---- Structure & binding (complex prediction) ----

    def estimate_cost(self, input_payload: dict, model: str = "boltz-2.1") -> dict:
        """Get USD estimate for a single structure_and_binding request."""
        resp = self.client.predictions.structure_and_binding.estimate_cost(
            input=input_payload,
            model=model,
        )
        return _to_dict(resp)

    def start_prediction(
        self,
        input_payload: dict,
        model: str = "boltz-2.1",
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Submit a structure_and_binding job. Returns immediately with id+status."""
        kwargs: dict[str, Any] = dict(input=input_payload, model=model)
        if idempotency_key:
            kwargs["idempotency_key"] = idempotency_key
        resp = self.client.predictions.structure_and_binding.start(**kwargs)
        return _to_dict(resp)

    def retrieve(self, prediction_id: str) -> dict:
        """Poll a previously-submitted prediction by id.

        The returned dict contains status (queued|running|completed|failed) and,
        when completed, the per-sample structure outputs and confidence scores
        (ipTM, plDDT, etc.). Prediction Agent uses this in a bounded poll loop.
        """
        resp = self.client.predictions.structure_and_binding.retrieve(id=prediction_id)
        return _to_dict(resp)

    # ---- Protein binder design (kept for completeness; PD-L1 demo doesn't use this) ----

    def design_estimate_cost(self, **kwargs) -> dict:
        resp = self.client.protein.design.estimate_cost(**kwargs)
        return _to_dict(resp)

    def design_start(self, **kwargs) -> dict:
        resp = self.client.protein.design.start(**kwargs)
        return _to_dict(resp)


# ----------------------------------------------------------------------
# Payload constructors
# ----------------------------------------------------------------------


def build_complex_input(
    target_seq: str,
    peptide_seq: str,
    peptide_chain: str = "B",
    target_chain: str = "A",
    num_samples: int = 1,
    target_msa: str | dict = "auto",
    peptide_msa: str | dict | None = None,
) -> dict:
    """Construct an Input payload for the Boltz API structure_and_binding endpoint.

    # v2 fix: the v1 build used {id/sequence/msa-string} which the SDK
    # rejected with a Pydantic validation error. The correct shape (verified
    # live 2026-06-27) is {chain_ids/type/value/msa-dict-or-omitted}.

    Parameters
    ----------
    target_seq:
        One-letter amino acid sequence for the target protein.
    peptide_seq:
        One-letter amino acid sequence for the peptide binder.
    peptide_chain, target_chain:
        Chain IDs to assign. Conventionally A=target, B=peptide.
    num_samples:
        Number of independent seeds the server runs per call. >1 enables
        the self_consistency Critic layer downstream.
    target_msa:
        "auto" (default; server-side AlphaFold MSA) or "empty" (skip MSA;
        much faster, used in the smoke tests) or a custom MSA dict.
    peptide_msa:
        Default None → omit msa entirely on the peptide entity (server falls
        back to auto). The live validated payload (L623) used {"type": "empty"}
        explicitly to skip MSA on the short peptide chain.

    Returns
    -------
    dict suitable for `BoltzAPIWrapper.start_prediction(input_payload=...)`.
    """
    target_entity: dict[str, Any] = {
        "chain_ids": [target_chain],
        "type": "protein",
        "value": target_seq,
    }
    if target_msa == "auto":
        # Omit msa on protein entities → server uses automatic MSA generation.
        pass
    elif target_msa == "empty":
        target_entity["msa"] = {"type": "empty"}
    elif isinstance(target_msa, dict):
        target_entity["msa"] = target_msa

    peptide_entity: dict[str, Any] = {
        "chain_ids": [peptide_chain],
        "type": "protein",
        "value": peptide_seq,
    }
    if peptide_msa is None:
        # Default for short peptides: explicit empty (matches validated payload).
        peptide_entity["msa"] = {"type": "empty"}
    elif peptide_msa == "auto":
        pass
    elif peptide_msa == "empty":
        peptide_entity["msa"] = {"type": "empty"}
    elif isinstance(peptide_msa, dict):
        peptide_entity["msa"] = peptide_msa

    return {
        "entities": [target_entity, peptide_entity],
        "binding": {
            "type": "protein_protein_binding",
            "binder_chain_ids": [peptide_chain],
        },
        "num_samples": int(num_samples),
    }


# ----------------------------------------------------------------------
# Response helpers
# ----------------------------------------------------------------------


def _to_dict(resp: Any) -> dict:
    """Normalize Pydantic SDK responses (or raw dicts in tests) to plain dicts."""
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    try:
        return dict(resp)
    except Exception:
        return {"_raw": str(resp)}



# ----------------------------------------------------------------------
# USD extractor (canonical location; agents/prediction.py re-exports this)
# ----------------------------------------------------------------------
# v2 fix: string-form cost responses ("0.0500") now parsed as floats.
# The original in-agent copy of this only accepted int/float and silently
# dropped the cap; this consolidated version is the source of truth.

def _extract_usd(cost_response: dict) -> Optional[float]:
    """Best-effort USD extraction from a Boltz estimate_cost response.

    Handles top-level keys (usd / total_usd / estimated_cost_usd / cost_usd /
    amount_usd / estimated_cost), nested {credits, usd}, and string-form
    values ("0.0500") which the live API returns.
    """
    if cost_response is None:
        return None
    for key in ("usd", "total_usd", "estimated_cost_usd", "estimated_cost",
                "amount_usd", "cost_usd"):
        v = cost_response.get(key)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    # Nested {cost: {usd: ..., credits: ...}}
    for key in ("cost", "amount", "estimate"):
        v = cost_response.get(key)
        if isinstance(v, dict):
            for inner in ("usd", "total_usd", "estimated_cost_usd"):
                inner_v = v.get(inner)
                if isinstance(inner_v, (int, float)):
                    return float(inner_v)
                if isinstance(inner_v, str):
                    try:
                        return float(inner_v)
                    except (TypeError, ValueError):
                        continue
    return None
