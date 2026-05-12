from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from .denario_adapter import load_denario_prompt_hint
from .models import AgentVote, CellScore, DebateTurn, RankedCell
from .settings import get_model, get_secret
from .telemetry import optional_observe as observe


class DebateState(TypedDict):
    cell: CellScore
    votes: list[AgentVote]
    transcript: list[DebateTurn]
    summary: str


MODEL_NAME = get_model("OPENAI_DEBATE_MODEL", "gpt-4o-mini")
REQUIRE_LLM_DEBATE = os.getenv("OPENAI_DEBATE_REQUIRED", "0") != "0"

AGENT_SPECS = {
    "HeatAgent": {
        "focus": "urban heat mitigation and cooling impact",
        "score_field": "cooling_need",
        "resources": [
            "resource://scores.cooling_need",
            "resource://scores.combined_score",
            "resource://evidence.nearby_buildings",
            "resource://evidence.nearby_trees",
        ],
        "system_prompt": (
            "You are HeatAgent in an urban climate planning debate. "
            "Prioritize heat mitigation and the pressure created by dense built form and weak canopy relief. "
            "You are participating in a shared transcript with other expert agents. "
            "Return compact JSON with keys stance, score, rationale, message."
        ),
    },
    "PollutionAgent": {
        "focus": "air-quality improvement and traffic-exposure reduction",
        "score_field": "pollution_need",
        "resources": [
            "resource://scores.pollution_need",
            "resource://scores.combined_score",
            "resource://evidence.nearby_road_nodes",
            "resource://evidence.pollution_pressure_proxy",
            "resource://evidence.nearby_trees",
        ],
        "system_prompt": (
            "You are PollutionAgent in an urban climate planning debate. "
            "Prioritize interventions that reduce local pollution exposure and road-adjacent urban stress. "
            "You are participating in a shared transcript with other expert agents. "
            "Return compact JSON with keys stance, score, rationale, message."
        ),
    },
    "EquityAgent": {
        "focus": "environmental equity and access to cooling and green space",
        "score_field": "equity_need",
        "resources": [
            "resource://scores.equity_need",
            "resource://scores.combined_score",
            "resource://evidence.park_overlap",
            "resource://evidence.nearby_trees",
        ],
        "system_prompt": (
            "You are EquityAgent in an urban climate planning debate. "
            "Prioritize underserved areas, lack of park access, and distributional fairness. "
            "You are participating in a shared transcript with other expert agents. "
            "Return compact JSON with keys stance, score, rationale, message."
        ),
    },
    "BiodiversityAgent": {
        "focus": "habitat connectivity and ecological uplift",
        "score_field": "biodiversity_need",
        "resources": [
            "resource://scores.biodiversity_need",
            "resource://scores.combined_score",
            "resource://evidence.park_overlap",
            "resource://evidence.nearby_trees",
        ],
        "system_prompt": (
            "You are BiodiversityAgent in an urban climate planning debate. "
            "Prioritize ecological connectivity, habitat support, and long-term green network value. "
            "You are participating in a shared transcript with other expert agents. "
            "Return compact JSON with keys stance, score, rationale, message."
        ),
    },
    "FeasibilityAgent": {
        "focus": "implementation feasibility and budget realism",
        "score_field": "feasibility_blend",
        "resources": [
            "resource://scores.feasibility",
            "resource://scores.budget_fit",
            "resource://scores.feasibility_blend",
            "resource://evidence.nearby_road_nodes",
            "resource://evidence.centroid_blocked_by_building",
            "resource://evidence.centroid_blocked_by_park",
        ],
        "system_prompt": (
            "You are FeasibilityAgent in an urban climate planning debate. "
            "Prioritize whether the site is practical to deliver and worth the implementation cost. "
            "You are participating in a shared transcript with other expert agents. "
            "Return compact JSON with keys stance, score, rationale, message."
        ),
    },
    "CritiqueAgent": {
        "focus": "stress-testing the recommendation by exposing contradictions, weak assumptions, and unresolved risks",
        "score_field": "risk_adjusted_assessment",
        "resources": [
            "resource://scores.cooling_need",
            "resource://scores.pollution_need",
            "resource://scores.equity_need",
            "resource://scores.biodiversity_need",
            "resource://scores.feasibility",
            "resource://scores.budget_fit",
            "resource://scores.combined_score",
            "resource://evidence.park_overlap",
            "resource://evidence.nearby_road_nodes",
            "resource://evidence.pollution_pressure_proxy",
            "resource://evidence.centroid_blocked_by_building",
            "resource://evidence.centroid_blocked_by_park",
        ],
        "system_prompt": (
            "You are CritiqueAgent in an urban climate planning debate. "
            "Your job is to challenge overconfidence, expose weak assumptions, and pressure-test the other agents' claims. "
            "You are participating in a shared transcript with other expert agents. "
            "Return compact JSON with keys stance, score, rationale, message."
        ),
    },
}


def _stance(score: float) -> str:
    if score >= 0.65:
        return "support"
    if score <= 0.35:
        return "oppose"
    return "abstain"


def _stance_phrase(stance: str) -> str:
    if stance == "support":
        return "support"
    if stance == "oppose":
        return "oppose"
    return "remain cautious about"


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI | None:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, timeout=8.0, max_retries=0)


def _debate_payload(cell: CellScore) -> dict[str, object]:
    feasibility_blend = round((cell.feasibility + cell.budget_fit) / 2.0, 4)
    return {
        "cell_id": cell.cell_id,
        "scores": {
            "cooling_need": cell.cooling_need,
            "pollution_need": cell.pollution_need,
            "equity_need": cell.equity_need,
            "biodiversity_need": cell.biodiversity_need,
            "feasibility": cell.feasibility,
            "budget_fit": cell.budget_fit,
            "feasibility_blend": feasibility_blend,
            "combined_score": cell.combined_score,
        },
        "evidence": cell.evidence,
    }


def _agent_resource_packet(cell: CellScore, agent_name: str) -> dict[str, object]:
    feasibility_blend = round((cell.feasibility + cell.budget_fit) / 2.0, 4)
    packets = {
        "HeatAgent": {
            "scores": {
                "cooling_need": cell.cooling_need,
                "combined_score": cell.combined_score,
            },
            "evidence": {
                "nearby_buildings": cell.evidence["nearby_buildings"],
                "nearby_trees": cell.evidence["nearby_trees"],
            },
        },
        "PollutionAgent": {
            "scores": {
                "pollution_need": cell.pollution_need,
                "combined_score": cell.combined_score,
            },
            "evidence": {
                "nearby_road_nodes": cell.evidence["nearby_road_nodes"],
                "pollution_pressure_proxy": cell.evidence["pollution_pressure_proxy"],
                "nearby_trees": cell.evidence["nearby_trees"],
            },
        },
        "EquityAgent": {
            "scores": {
                "equity_need": cell.equity_need,
                "combined_score": cell.combined_score,
            },
            "evidence": {
                "park_overlap": cell.evidence["park_overlap"],
                "nearby_trees": cell.evidence["nearby_trees"],
            },
        },
        "BiodiversityAgent": {
            "scores": {
                "biodiversity_need": cell.biodiversity_need,
                "combined_score": cell.combined_score,
            },
            "evidence": {
                "park_overlap": cell.evidence["park_overlap"],
                "nearby_trees": cell.evidence["nearby_trees"],
            },
        },
        "FeasibilityAgent": {
            "scores": {
                "feasibility": cell.feasibility,
                "budget_fit": cell.budget_fit,
                "feasibility_blend": feasibility_blend,
            },
            "evidence": {
                "nearby_road_nodes": cell.evidence["nearby_road_nodes"],
                "centroid_blocked_by_building": cell.evidence["centroid_blocked_by_building"],
                "centroid_blocked_by_park": cell.evidence["centroid_blocked_by_park"],
            },
        },
        "CritiqueAgent": {
            "scores": {
                "cooling_need": cell.cooling_need,
                "pollution_need": cell.pollution_need,
                "equity_need": cell.equity_need,
                "biodiversity_need": cell.biodiversity_need,
                "feasibility": cell.feasibility,
                "budget_fit": cell.budget_fit,
                "combined_score": cell.combined_score,
            },
            "evidence": {
                "park_overlap": cell.evidence["park_overlap"],
                "nearby_road_nodes": cell.evidence["nearby_road_nodes"],
                "pollution_pressure_proxy": cell.evidence["pollution_pressure_proxy"],
                "centroid_blocked_by_building": cell.evidence["centroid_blocked_by_building"],
                "centroid_blocked_by_park": cell.evidence["centroid_blocked_by_park"],
            },
        },
    }
    return {
        "resource_ids": AGENT_SPECS[agent_name]["resources"],
        "resource_packet": packets[agent_name],
    }


def _serialize_votes(votes: list[AgentVote]) -> list[dict[str, object]]:
    return [
        {
            "agent": vote.agent,
            "stance": vote.stance,
            "score": vote.score,
            "rationale": vote.rationale,
            "preferred_cell_id": vote.preferred_cell_id,
        }
        for vote in votes
    ]


def _serialize_transcript(transcript: list[DebateTurn]) -> list[dict[str, str]]:
    return [
        {
            "agent": turn.agent,
            "phase": turn.phase,
            "content": turn.content,
        }
        for turn in transcript
    ]


def _upsert_vote(votes: list[AgentVote], vote: AgentVote) -> list[AgentVote]:
    updated = [existing for existing in votes if existing.agent != vote.agent]
    updated.append(vote)
    return updated


def _other_turns(transcript: list[DebateTurn], agent_name: str) -> list[DebateTurn]:
    return [turn for turn in transcript if turn.agent not in {agent_name, "Judge"}]


def _latest_peer_name(transcript: list[DebateTurn], agent_name: str) -> str | None:
    turns = _other_turns(transcript, agent_name)
    if not turns:
        return None
    return turns[-1].agent


def _phase_instruction(agent_name: str, phase: str) -> str:
    if agent_name == "CritiqueAgent" and phase == "opening":
        return (
            "This is the opening critique round. Identify the biggest weak assumption or unresolved contradiction in the case so far. "
            "Speak in natural language and directly reference at least one earlier agent by name."
        )
    if agent_name == "CritiqueAgent" and phase == "rebuttal":
        return (
            "This is the closing critique round. Decide whether the other agents actually answered the core weaknesses you raised. "
            "Speak in natural language and directly reference at least one other agent by name."
        )
    if phase == "opening":
        return (
            "This is the opening round. Add an initial argument to the shared debate transcript in natural language. "
            "If other opening statements already exist, you may acknowledge them by name, but focus on stating your own case clearly."
        )
    return (
        "This is the rebuttal round. Read the existing transcript and explicitly respond to at least one other agent by name. "
        "Challenge or support a concrete claim from the discussion rather than just restating your position. "
        "You may keep or revise your score and stance after considering the discussion."
    )


def _fallback_turn(
    cell: CellScore,
    transcript: list[DebateTurn],
    agent_name: str,
    phase: str,
    error: str | None = None,
) -> tuple[AgentVote, DebateTurn]:
    latest_peer = _latest_peer_name(transcript, agent_name) or "the rest of the panel"
    all_peers = sorted({turn.agent for turn in transcript if turn.agent not in {agent_name, "Judge"}})
    peer_list = ", ".join(all_peers) if all_peers else "the panel"

    if agent_name == "HeatAgent":
        score = cell.cooling_need
        stance = _stance(score)
        opening_message = (
            f"HeatAgent: I {_stance_phrase(stance)} this site because the cooling need is {score:.2f}, "
            f"with {cell.evidence['nearby_buildings']} nearby buildings driving heat pressure."
        )
        rebuttal_message = (
            f"HeatAgent rebuttal: {latest_peer} is right to raise trade-offs, but I still {_stance_phrase(stance)} this site "
            "because the local thermal burden remains unusually high."
        )
        rationale = (
            f"Fallback heuristic used because the model call was unavailable"
            f"{f' ({error})' if error else ''}. "
            f"Cooling priority is {score:.2f} with {cell.evidence['nearby_buildings']} nearby buildings and {cell.evidence['nearby_trees']} nearby trees."
        )
    elif agent_name == "PollutionAgent":
        score = cell.pollution_need
        stance = _stance(score)
        opening_message = (
            f"PollutionAgent: I {_stance_phrase(stance)} this site because pollution need is {score:.2f}; "
            f"road exposure is {cell.evidence['nearby_road_nodes']} and the pollution pressure proxy is {cell.evidence['pollution_pressure_proxy']}."
        )
        rebuttal_message = (
            f"PollutionAgent rebuttal: I hear {latest_peer}, but I still {_stance_phrase(stance)} this site because roadside exposure "
            "makes canopy intervention especially valuable here."
        )
        rationale = (
            f"Fallback heuristic used because the model call was unavailable"
            f"{f' ({error})' if error else ''}. "
            f"Pollution reduction priority is {score:.2f}, driven by road exposure={cell.evidence['nearby_road_nodes']} and urban pressure proxy={cell.evidence['pollution_pressure_proxy']}."
        )
    elif agent_name == "EquityAgent":
        score = cell.equity_need
        stance = _stance(score)
        opening_message = (
            f"EquityAgent: I {_stance_phrase(stance)} this site because equity need is {score:.2f}; park overlap is "
            f"{cell.evidence['park_overlap']} and local tree scarcity still matters."
        )
        rebuttal_message = (
            f"EquityAgent rebuttal: {latest_peer} makes a fair point, but I still {_stance_phrase(stance)} this site because "
            "access to green relief should stay central to the decision."
        )
        rationale = (
            f"Fallback heuristic used because the model call was unavailable"
            f"{f' ({error})' if error else ''}. "
            f"Equity need is {score:.2f}; the cell has park overlap={cell.evidence['park_overlap']} and tree scarcity={cell.evidence['nearby_trees']}."
        )
    elif agent_name == "BiodiversityAgent":
        score = cell.biodiversity_need
        stance = _stance(score)
        opening_message = (
            f"BiodiversityAgent: I {_stance_phrase(stance)} this site because biodiversity signal is {score:.2f}, "
            "so the ecological upside is meaningful but not overwhelming."
        )
        rebuttal_message = (
            f"BiodiversityAgent rebuttal: I understand {latest_peer}, but I still {_stance_phrase(stance)} this site because "
            "connectivity and habitat value should remain part of the trade-off."
        )
        rationale = (
            f"Fallback heuristic used because the model call was unavailable"
            f"{f' ({error})' if error else ''}. "
            f"Biodiversity signal is {score:.2f}, balancing current tree adjacency and park connectivity."
        )
    elif agent_name == "FeasibilityAgent":
        score = round((cell.feasibility + cell.budget_fit) / 2.0, 4)
        stance = _stance(score)
        opening_message = (
            f"FeasibilityAgent: I {_stance_phrase(stance)} this site for delivery because the blended feasibility score is {score:.2f}, "
            f"with feasibility={cell.feasibility:.2f} and budget fit={cell.budget_fit:.2f}."
        )
        rebuttal_message = (
            f"FeasibilityAgent rebuttal: I understand why {latest_peer} sees upside, but I still {_stance_phrase(stance)} this site "
            "because implementation constraints can outweigh theoretical benefits."
        )
        rationale = (
            f"Fallback heuristic used because the model call was unavailable"
            f"{f' ({error})' if error else ''}. "
            f"Feasibility averages to {score:.2f} from placement feasibility={cell.feasibility:.2f} and budget fit={cell.budget_fit:.2f}."
        )
    else:
        score = round(
            (
                cell.combined_score
                + min(
                    cell.feasibility,
                    cell.budget_fit,
                    cell.equity_need,
                    cell.biodiversity_need,
                    cell.pollution_need,
                )
            )
            / 2.0,
            4,
        )
        stance = _stance(score)
        opening_message = (
            f"CritiqueAgent: HeatAgent and PollutionAgent both make strong upside cases, but I {_stance_phrase(stance)} this recommendation "
            "because the weaker feasibility, equity, or biodiversity signals suggest the headline score may be overconfident."
        )
        rebuttal_message = (
            f"CritiqueAgent rebuttal: After hearing {peer_list}, I still {_stance_phrase(stance)} the recommendation because the panel "
            "has not fully resolved whether the practical and distributional weaknesses are acceptable trade-offs."
        )
        rationale = (
            f"Fallback heuristic used because the model call was unavailable"
            f"{f' ({error})' if error else ''}. "
            f"Critique review discounted the overall case to {score:.2f} because strong upside remains in tension with weaker feasibility, equity, biodiversity, or pollution trade-offs."
        )

    vote = AgentVote(
        agent=agent_name,
        stance=stance,
        score=round(score, 4),
        rationale=rationale,
    )
    turn = DebateTurn(
        agent=agent_name,
        phase=phase,
        content=opening_message if phase == "opening" else rebuttal_message,
    )
    return vote, turn


def _llm_turn(
    cell: CellScore,
    transcript: list[DebateTurn],
    votes: list[AgentVote],
    agent_name: str,
    phase: str,
) -> tuple[AgentVote, DebateTurn]:
    client = _get_openai_client()
    if client is None:
        if REQUIRE_LLM_DEBATE:
            raise RuntimeError(
                f"{agent_name} could not vote because OPENAI_API_KEY is missing and OPENAI_DEBATE_REQUIRED=1."
            )
        return _fallback_turn(cell, transcript, agent_name, phase, error="missing OPENAI_API_KEY")

    spec = AGENT_SPECS[agent_name]
    payload = {
        "cell": _debate_payload(cell),
        "agent_resources": _agent_resource_packet(cell, agent_name),
        "current_votes": _serialize_votes(votes),
        "transcript": _serialize_transcript(transcript),
    }
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": spec["system_prompt"],
                },
                {
                    "role": "user",
                    "content": (
                        f"You are speaking during the {phase} round of a shared urban climate debate.\n"
                        f"Primary score field: {spec['score_field']}\n"
                        f"Role focus: {spec['focus']}\n"
                        f"Assigned resources: {', '.join(spec['resources'])}\n"
                        f"{_phase_instruction(agent_name, phase)}\n"
                        "Use only the assigned resource packet and the shared transcript.\n"
                        "Return strict JSON with keys stance, score, rationale, message.\n"
                        "Score must be between 0 and 1. Stance must be support, abstain, or oppose.\n"
                        "The message should be one short paragraph written as something this agent would actually say to the other agents.\n"
                        "Use direct, natural language rather than labels, bullet points, or report-style phrasing.\n"
                        "In the rebuttal round, mention at least one other agent by name.\n"
                        f"Debate state: {json.dumps(payload)}"
                    ),
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        score = max(0.0, min(1.0, float(parsed["score"])))
        stance = str(parsed["stance"]).strip().lower()
        if stance not in {"support", "abstain", "oppose"}:
            stance = _stance(score)
        rationale = str(parsed["rationale"]).strip()
        message = str(parsed["message"]).strip()
        if not rationale:
            raise ValueError("empty rationale")
        if not message:
            raise ValueError("empty message")
        vote = AgentVote(
            agent=agent_name,
            stance=stance,
            score=round(score, 4),
            rationale=rationale,
        )
        turn = DebateTurn(agent=agent_name, phase=phase, content=message)
        return vote, turn
    except Exception as exc:
        if REQUIRE_LLM_DEBATE:
            raise RuntimeError(f"{agent_name} model call failed: {type(exc).__name__}: {exc}") from exc
        return _fallback_turn(cell, transcript, agent_name, phase, error=f"{type(exc).__name__}: {exc}")


def _run_agent_turn(state: DebateState, agent_name: str, phase: str) -> DebateState:
    vote, turn = _llm_turn(state["cell"], state["transcript"], state["votes"], agent_name, phase)
    return {
        "cell": state["cell"],
        "votes": _upsert_vote(state["votes"], vote),
        "transcript": [*state["transcript"], turn],
        "summary": state["summary"],
    }


def _make_agent_node(agent_name: str, phase: str):
    @observe(name=f"{agent_name.lower()}_{phase}")
    def node(state: DebateState) -> DebateState:
        return _run_agent_turn(state, agent_name, phase)

    return node


heat_opening = _make_agent_node("HeatAgent", "opening")
pollution_opening = _make_agent_node("PollutionAgent", "opening")
equity_opening = _make_agent_node("EquityAgent", "opening")
biodiversity_opening = _make_agent_node("BiodiversityAgent", "opening")
feasibility_opening = _make_agent_node("FeasibilityAgent", "opening")
critique_opening = _make_agent_node("CritiqueAgent", "opening")
heat_rebuttal = _make_agent_node("HeatAgent", "rebuttal")
pollution_rebuttal = _make_agent_node("PollutionAgent", "rebuttal")
equity_rebuttal = _make_agent_node("EquityAgent", "rebuttal")
biodiversity_rebuttal = _make_agent_node("BiodiversityAgent", "rebuttal")
feasibility_rebuttal = _make_agent_node("FeasibilityAgent", "rebuttal")
critique_rebuttal = _make_agent_node("CritiqueAgent", "rebuttal")


@observe(name="judge_vote")
def judge(state: DebateState) -> DebateState:
    prompt_hint = load_denario_prompt_hint()
    cell = state["cell"]
    votes = state["votes"]
    transcript = state["transcript"]
    final_score = sum(v.score for v in votes) / max(len(votes), 1)
    supports = sum(v.stance == "support" for v in votes)
    agent_turns = sum(turn.agent != "Judge" for turn in transcript)
    summary = (
        f"Judge summary for {cell.cell_id}: final debate score {final_score:.2f}. "
        f"{supports}/{len(votes)} agents supported the cell after {agent_turns} shared debate turns. "
        f"Cooling={cell.cooling_need:.2f}, pollution={cell.pollution_need:.2f}, equity={cell.equity_need:.2f}, biodiversity={cell.biodiversity_need:.2f}. "
        f"Denario prompt lineage: '{prompt_hint}'"
    )
    judge_turn = DebateTurn(
        agent="Judge",
        phase="summary",
        content=(
            f"Judge: I averaged the final agent scores to {final_score:.2f}. "
            f"{supports}/{len(votes)} agents support the site after reviewing the shared transcript."
        ),
    )
    return {
        "cell": cell,
        "votes": votes,
        "transcript": [*transcript, judge_turn],
        "summary": summary,
    }


def build_debate_graph():
    graph = StateGraph(DebateState)
    graph.add_node("heat_opening", heat_opening)
    graph.add_node("pollution_opening", pollution_opening)
    graph.add_node("equity_opening", equity_opening)
    graph.add_node("biodiversity_opening", biodiversity_opening)
    graph.add_node("feasibility_opening", feasibility_opening)
    graph.add_node("critique_opening", critique_opening)
    graph.add_node("heat_rebuttal", heat_rebuttal)
    graph.add_node("pollution_rebuttal", pollution_rebuttal)
    graph.add_node("equity_rebuttal", equity_rebuttal)
    graph.add_node("biodiversity_rebuttal", biodiversity_rebuttal)
    graph.add_node("feasibility_rebuttal", feasibility_rebuttal)
    graph.add_node("critique_rebuttal", critique_rebuttal)
    graph.add_node("judge", judge)
    graph.add_edge(START, "heat_opening")
    graph.add_edge("heat_opening", "pollution_opening")
    graph.add_edge("pollution_opening", "equity_opening")
    graph.add_edge("equity_opening", "biodiversity_opening")
    graph.add_edge("biodiversity_opening", "feasibility_opening")
    graph.add_edge("feasibility_opening", "critique_opening")
    graph.add_edge("critique_opening", "heat_rebuttal")
    graph.add_edge("heat_rebuttal", "pollution_rebuttal")
    graph.add_edge("pollution_rebuttal", "equity_rebuttal")
    graph.add_edge("equity_rebuttal", "biodiversity_rebuttal")
    graph.add_edge("biodiversity_rebuttal", "feasibility_rebuttal")
    graph.add_edge("feasibility_rebuttal", "critique_rebuttal")
    graph.add_edge("critique_rebuttal", "judge")
    graph.add_edge("judge", END)
    return graph.compile()


def rank_with_debate(
    cells: list[CellScore],
    limit: int = 8,
    on_cell_done: Callable[[int, int, str], None] | None = None,
) -> list[RankedCell]:
    app = build_debate_graph()
    ranked: list[RankedCell] = []
    total = min(len(cells), limit)
    for i, cell in enumerate(cells[:limit]):
        state = app.invoke({"cell": cell, "votes": [], "transcript": [], "summary": ""})
        votes = state["votes"]
        final_score = sum(v.score for v in votes) / max(len(votes), 1)
        ranked.append(
            RankedCell(
                cell=cell,
                votes=votes,
                transcript=state["transcript"],
                final_score=round(final_score, 4),
                summary=state["summary"],
            )
        )
        if on_cell_done:
            on_cell_done(i + 1, total, cell.cell_id)
    ranked.sort(key=lambda item: item.final_score, reverse=True)
    return ranked
