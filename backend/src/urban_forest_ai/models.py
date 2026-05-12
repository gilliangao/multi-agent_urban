from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CellScore:
    cell_id: str
    centroid_x: float
    centroid_y: float
    cooling_need: float
    pollution_need: float
    equity_need: float
    biodiversity_need: float
    feasibility: float
    budget_fit: float
    combined_score: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentVote:
    agent: str
    stance: str
    score: float
    rationale: str
    preferred_cell_id: str | None = None


@dataclass(slots=True)
class DebateTurn:
    agent: str
    phase: str
    content: str


@dataclass(slots=True)
class RankedCell:
    cell: CellScore
    votes: list[AgentVote]
    transcript: list[DebateTurn]
    final_score: float
    summary: str
    reason_chain: list[str] = field(default_factory=list)
    llm_explanation: str | None = None


@dataclass(slots=True)
class CoordinationStep:
    round_index: int
    selected_cell_id: str
    agent_preferences: dict[str, str]
    coalition_size: int
    coordination_score: float
    shared_reward: float
    rationale: str


@dataclass(slots=True)
class CoordinationEpisode:
    budget: int
    selected_cells: list[str]
    steps: list[CoordinationStep]
    final_shared_reward: float
    mean_coordination_score: float
    conflict_rate: float
    policy_summary: str
    indicator_summary: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    area_name: str
    denario_status: dict[str, str]
    candidates_considered: int
    ranked_cells: list[RankedCell]
    baseline_summary: dict[str, float]
    coordination_episode: CoordinationEpisode | None = None
    trace_path: str = ""
