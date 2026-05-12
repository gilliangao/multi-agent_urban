from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from .data_loader import load_buildings, load_parks, load_road_nodes, load_trees, select_area
from .debate import rank_with_debate
from .denario_adapter import detect_denario_runtime
from .explainability import build_reason_chain, maybe_generate_llm_explanation, write_trace
from .marl import align_ranked_cells_with_episode, run_coordination_episode
from .models import PipelineResult
from .scoring import make_grid, score_cells
from .telemetry import optional_observe as observe


@observe(name="urban_forest_pipeline")
def run_pipeline(
    area_query: str | None = None,
    level: str = "msoa",
    cell_size: float = 250.0,
    top_k: int = 5,
    budget: int = 3,
    on_stage: Callable[[str], None] | None = None,
    on_cell_done: Callable[[int, int, str], None] | None = None,
) -> PipelineResult:
    def _stage(msg: str) -> None:
        if on_stage:
            on_stage(msg)

    _stage("Loading spatial data...")
    area_name, aoi = select_area(area_query, level=level)
    parks = load_parks(aoi)
    roads = load_road_nodes(aoi)
    buildings = load_buildings(aoi)
    trees = load_trees(aoi)

    _stage("Generating and scoring candidate cells...")
    cells = make_grid(aoi, cell_size=cell_size)
    scored = score_cells(cells, buildings=buildings, trees=trees, parks=parks, road_nodes=roads)

    _stage("Running coordination episode...")
    coordination_episode = run_coordination_episode(scored[: max(top_k * 3, budget * 3)], budget=budget)

    debate_limit = max(top_k * 2, top_k)
    _stage(f"Running LLM debate for top {debate_limit} candidate cells...")
    debated = rank_with_debate(scored, limit=debate_limit, on_cell_done=on_cell_done)
    debated = align_ranked_cells_with_episode(debated, coordination_episode)
    top_ranked = debated[:top_k]

    _stage("Generating explanations...")
    for item in top_ranked:
        item.reason_chain = build_reason_chain(item)
        item.reason_chain.append(coordination_episode.policy_summary)
        item.llm_explanation = maybe_generate_llm_explanation(item)

    baseline_random = mean(cell.combined_score for cell in scored[-top_k:]) if scored else 0.0
    baseline_greedy = mean(cell.combined_score for cell in scored[:top_k]) if scored else 0.0
    debate_top = mean(item.final_score for item in top_ranked) if top_ranked else 0.0
    baseline_summary = {
        "greedy_feature_baseline": round(baseline_greedy, 4),
        "debate_baseline": round(debate_top, 4),
        "low_score_reference": round(baseline_random, 4),
    }
    trace_path = write_trace(area_name, baseline_summary, top_ranked, coordination_episode)

    return PipelineResult(
        area_name=area_name,
        denario_status=detect_denario_runtime(),
        candidates_considered=len(scored),
        ranked_cells=top_ranked,
        baseline_summary=baseline_summary,
        coordination_episode=coordination_episode,
        trace_path=trace_path,
    )
