from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
import json

from openai import OpenAI

from .models import CoordinationEpisode, RankedCell
from .settings import get_model, get_secret


def build_reason_chain(item: RankedCell) -> list[str]:
    cell = item.cell
    agent_turns = sum(turn.agent != "Judge" for turn in item.transcript)
    vote_summary = ", ".join(f"{vote.agent}={vote.stance}" for vote in item.votes)
    reasons = [
        f"Cell {cell.cell_id} started with a feature score of {cell.combined_score:.2f} and ended with a debate score of {item.final_score:.2f}.",
        f"Cooling need was {cell.cooling_need:.2f} based on {cell.evidence['nearby_buildings']} nearby buildings and {cell.evidence['nearby_trees']} nearby trees.",
        f"Pollution reduction need was {cell.pollution_need:.2f}, using road exposure and dense urban pressure as a local pollution proxy.",
        f"Equity need was {cell.equity_need:.2f} because park overlap={cell.evidence['park_overlap']} and local tree scarcity was evaluated from nearby tree count.",
        f"Biodiversity need was {cell.biodiversity_need:.2f}, reflecting how the cell connects to existing green structure.",
        f"Feasibility was {cell.feasibility:.2f} and budget fit was {cell.budget_fit:.2f}, with {cell.evidence['nearby_road_nodes']} nearby road nodes affecting implementation difficulty.",
        f"The agents exchanged {agent_turns} shared debate turns before the judge summary was added.",
        f"The final recommendation reflects votes from {vote_summary}.",
    ]
    return reasons


def maybe_generate_llm_explanation(item: RankedCell) -> str | None:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key, timeout=8.0, max_retries=0)
        model = get_model("OPENAI_EXPLAINER_MODEL", "gpt-4o-mini")
        prompt = {
            "cell_id": item.cell.cell_id,
            "feature_score": item.cell.combined_score,
            "debate_score": item.final_score,
            "evidence": item.cell.evidence,
            "votes": [asdict(vote) for vote in item.votes],
            "reason_chain": item.reason_chain,
        }
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You explain urban planning recommendations. Be concise, evidence-based, and explicitly mention trade-offs.",
                },
                {
                    "role": "user",
                    "content": f"Explain why this cell was recommended for tree planting, using only this evidence: {json.dumps(prompt)}",
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"LLM explanation unavailable: {type(exc).__name__}: {exc}"


def write_trace(
    area_name: str,
    baseline_summary: dict[str, float],
    ranked_cells: list[RankedCell],
    coordination_episode: CoordinationEpisode | None = None,
) -> str:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    safe_area = area_name.lower().replace(" ", "_")
    path = out_dir / f"{safe_area}_trace.json"
    payload = {
        "area_name": area_name,
        "baseline_summary": baseline_summary,
        "coordination_episode": asdict(coordination_episode) if coordination_episode else None,
        "ranked_cells": [
            {
                "cell": asdict(item.cell),
                "votes": [asdict(vote) for vote in item.votes],
                "transcript": [asdict(turn) for turn in item.transcript],
                "final_score": item.final_score,
                "summary": item.summary,
                "reason_chain": item.reason_chain,
                "llm_explanation": item.llm_explanation,
            }
            for item in ranked_cells
        ],
    }
    try:
        path.write_text(json.dumps(payload, indent=2))
        return str(path.resolve())
    except Exception as exc:
        return f"Trace write unavailable: {type(exc).__name__}: {exc}"
