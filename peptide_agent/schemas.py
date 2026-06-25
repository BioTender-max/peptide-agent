"""Typed Pydantic schemas — the message protocol between agents.

Every agent communicates by producing/consuming these objects. Free-form
strings are deliberately rare; the Critic operates on this structured layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------- Provenance primitives ----------

EvidenceTag = Literal["VERIFIED", "DERIVED", "SUBJECTIVE"]
SourceType = Literal["literature", "pdb", "uniprot", "tool_output", "agent_decision"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


class EvidenceCard(BaseModel):
    """An atomic, append-only, hash-addressable evidence record.

    Every claim that appears in any downstream artifact MUST cite at least one
    EvidenceCard via card_id. Cards without a source are forbidden.
    """

    card_id: str = Field(default_factory=lambda: _new_id("evid"))
    claim: str
    source_id: str  # citation index, PDB ID, UniProt accession, tool-call hash
    source_type: SourceType
    source_url: Optional[str] = None
    tag: EvidenceTag = "DERIVED"
    confidence: float = 0.5
    extracted_by: str  # agent name
    timestamp: datetime = Field(default_factory=_now)
    payload: dict = Field(default_factory=dict)

    # Provenance graph
    derived_from: list[str] = Field(default_factory=list)  # other card_ids
    supersedes: Optional[str] = None  # an older card this one corrects
    content_hash: Optional[str] = None  # filled in by the ledger on commit


# ---------- Planning ----------


class TaskNode(BaseModel):
    task_id: str = Field(default_factory=lambda: _new_id("task"))
    name: str
    agent: Literal[
        "planner", "research", "structure", "design", "prediction", "critic", "reporter"
    ]
    inputs: list[str] = Field(default_factory=list)  # task_ids it depends on
    success_criteria: str
    tools_allowed: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "vetoed", "superseded"] = "pending"
    superseded_by: Optional[str] = None
    estimated_cost: dict = Field(default_factory=dict)  # tokens, gpu_h, wallclock_s


class TaskPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: _new_id("plan"))
    brief: str
    nodes: list[TaskNode]
    created_at: datetime = Field(default_factory=_now)
    revisions: int = 0


# ---------- Target / structure artifacts ----------


class TargetBrief(BaseModel):
    target_id: str  # e.g., "PD-L1"
    uniprot: Optional[str] = None
    gene: Optional[str] = None
    organism: Optional[str] = None
    length: Optional[int] = None
    sequence: Optional[str] = None
    function_summary: str = ""
    interaction_partners: list[str] = Field(default_factory=list)
    known_binders: list[dict] = Field(default_factory=list)  # name, modality, IC50/Kd, evidence_ids
    reference_pdbs: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)  # supports every line above


class Hotspot(BaseModel):
    hotspot_id: str = Field(default_factory=lambda: _new_id("hs"))
    chain: str
    residue_number: int
    residue_aa: str
    role: Literal["anchor", "hub", "rim", "ambiguous"] = "ambiguous"
    bsa: Optional[float] = None  # buried surface area on partner binding
    conservation: Optional[float] = None  # 0-1
    supported_by_tools: list[str] = Field(default_factory=list)
    consensus_score: int = 0  # number of independent tools/sources agreeing
    evidence_ids: list[str] = Field(default_factory=list)


class EpitopeMap(BaseModel):
    target_id: str
    reference_pdb: str
    partner_chain: Optional[str] = None
    hotspots: list[Hotspot]
    summary: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class StructureProfile(BaseModel):
    target_id: str
    pdbs_loaded: list[str]
    chains: dict = Field(default_factory=dict)  # pdb -> {chain: length}
    epitope_map: Optional[EpitopeMap] = None
    notes: str = ""


# ---------- Design artifacts ----------

GeneratorName = Literal[
    "mutation_scan",
    "esm_if",
    "llm_conditional",
    "rfdiffusion_proteinmpnn",
    "boltzgen",
    "boltz_protein_design",
]

Modality = Literal["linear", "cyclic_disulfide", "cyclic_headtail", "stapled"]


class DesignProvenance(BaseModel):
    generator: GeneratorName
    parent_sequence: Optional[str] = None
    parent_pdb: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    epitope_hash: Optional[str] = None
    seed: Optional[int] = None
    timestamp: datetime = Field(default_factory=_now)


class Candidate(BaseModel):
    cand_id: str = Field(default_factory=lambda: _new_id("cand"))
    sequence: str
    modality: Modality = "linear"
    length: int
    design_provenance: DesignProvenance
    intended_hotspots: list[str] = Field(default_factory=list)  # Hotspot.hotspot_id
    design_rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["proposed", "filtered_out", "predicted", "scored", "rejected", "shortlisted"] = "proposed"
    filter_reason: Optional[str] = None


# ---------- Prediction & scoring ----------


class ComplexPrediction(BaseModel):
    pred_id: str = Field(default_factory=lambda: _new_id("pred"))
    cand_id: str
    predictor: Literal["boltz_api", "boltz_hpc", "chai_1", "alphafold"]
    seed: Optional[int] = None
    ipTM: Optional[float] = None
    pTM: Optional[float] = None
    pLDDT_interface: Optional[float] = None
    rmsd_to_template: Optional[float] = None
    cif_path: Optional[str] = None
    raw_metrics: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)
    evidence_ids: list[str] = Field(default_factory=list)


class ScoreCard(BaseModel):
    cand_id: str
    structural: dict = Field(default_factory=dict)  # mean_ipTM, mean_pTM, ...
    interface: dict = Field(default_factory=dict)   # contact_count, BSA, hotspot_coverage
    energy_proxy: dict = Field(default_factory=dict)
    consistency: dict = Field(default_factory=dict)  # ensemble RMSF, chai_vs_boltz ipTM gap
    composite_score: float = 0.0
    confidence_class: Literal["high", "medium", "low", "rejected"] = "low"
    reasons: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


# ---------- Critic ----------


CriticLayer = Literal["evidence_gate", "cross_tool", "self_consistency", "calibrated_rejection"]


class Issue(BaseModel):
    layer: CriticLayer
    severity: Literal["info", "warn", "error"]
    message: str
    suggested_action: Optional[str] = None


class CriticReport(BaseModel):
    report_id: str = Field(default_factory=lambda: _new_id("crit"))
    target_agent: str
    target_artifact_id: str
    layers_run: list[CriticLayer]
    issues: list[Issue]
    verdict: Literal["pass", "warn", "veto"]
    recommended_action: Optional[str] = None
    timestamp: datetime = Field(default_factory=_now)
