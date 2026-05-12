from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the urban forest debate prototype.")
    parser.add_argument("--area", default=None, help="Substring match against LSOA21NM.")
    parser.add_argument("--cell-size", type=float, default=250.0, help="Grid cell size in projected meters.")
    parser.add_argument("--top-k", type=int, default=5, help="How many final cells to return.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(area_query=args.area, cell_size=args.cell_size, top_k=args.top_k)
    payload = {
        "area_name": result.area_name,
        "denario_status": result.denario_status,
        "candidates_considered": result.candidates_considered,
        "baseline_summary": result.baseline_summary,
        "coordination_episode": None
        if result.coordination_episode is None
        else {
            "budget": result.coordination_episode.budget,
            "selected_cells": result.coordination_episode.selected_cells,
            "final_shared_reward": result.coordination_episode.final_shared_reward,
            "mean_coordination_score": result.coordination_episode.mean_coordination_score,
            "conflict_rate": result.coordination_episode.conflict_rate,
            "indicator_summary": result.coordination_episode.indicator_summary,
            "policy_summary": result.coordination_episode.policy_summary,
            "steps": [
                {
                    "round_index": step.round_index,
                    "selected_cell_id": step.selected_cell_id,
                    "agent_preferences": step.agent_preferences,
                    "coalition_size": step.coalition_size,
                    "coordination_score": step.coordination_score,
                    "shared_reward": step.shared_reward,
                    "rationale": step.rationale,
                }
                for step in result.coordination_episode.steps
            ],
        },
        "trace_path": result.trace_path,
        "ranked_cells": [
            {
                "cell_id": item.cell.cell_id,
                "centroid": [round(item.cell.centroid_x, 2), round(item.cell.centroid_y, 2)],
                "feature_score": item.cell.combined_score,
                "debate_score": item.final_score,
                "evidence": item.cell.evidence,
                "votes": [asdict(vote) for vote in item.votes],
                "transcript": [asdict(turn) for turn in item.transcript],
                "summary": item.summary,
                "reason_chain": item.reason_chain,
                "llm_explanation": item.llm_explanation,
            }
            for item in result.ranked_cells
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
