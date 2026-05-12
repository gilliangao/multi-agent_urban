"""Lightweight MARL-style coordination helpers used by the planning pipeline."""

from __future__ import annotations

from statistics import mean

from ..models import CellScore, CoordinationEpisode, CoordinationStep, RankedCell


AGENT_NAMES = ["HeatAgent", "PollutionAgent", "EquityAgent", "BiodiversityAgent", "FeasibilityAgent"]


def _agent_priority(cell: CellScore, agent: str) -> float:
    if agent == "HeatAgent":
        return cell.cooling_need
    if agent == "PollutionAgent":
        return cell.pollution_need
    if agent == "EquityAgent":
        return cell.equity_need
    if agent == "BiodiversityAgent":
        return cell.biodiversity_need
    if agent == "FeasibilityAgent":
        return (cell.feasibility + cell.budget_fit) / 2.0
    raise ValueError(f"Unknown agent: {agent}")


def _rank_preferences(candidates: list[CellScore], remaining: set[str], agent: str) -> list[CellScore]:
    filtered = [cell for cell in candidates if cell.cell_id in remaining]
    return sorted(filtered, key=lambda cell: _agent_priority(cell, agent), reverse=True)


def run_coordination_episode(candidates: list[CellScore], budget: int = 3) -> CoordinationEpisode:
    """A lightweight MARL-style coordination episode over sequential planting choices."""

    remaining = {cell.cell_id for cell in candidates}
    by_id = {cell.cell_id: cell for cell in candidates}
    steps: list[CoordinationStep] = []
    selected_cells: list[str] = []

    for round_index in range(1, budget + 1):
        if not remaining:
            break

        preferences = {}
        for agent in AGENT_NAMES:
            ranked = _rank_preferences(candidates, remaining, agent)
            if ranked:
                preferences[agent] = ranked[0].cell_id

        vote_counts: dict[str, int] = {}
        for cell_id in preferences.values():
            vote_counts[cell_id] = vote_counts.get(cell_id, 0) + 1

        selected_cell_id = sorted(
            vote_counts.items(),
            key=lambda item: (
                item[1],
                by_id[item[0]].combined_score,
                by_id[item[0]].feasibility,
            ),
            reverse=True,
        )[0][0]
        selected = by_id[selected_cell_id]

        coalition_size = vote_counts[selected_cell_id]
        coordination_score = coalition_size / len(AGENT_NAMES)
        shared_reward = (
            0.28 * selected.cooling_need
            + 0.24 * selected.pollution_need
            + 0.18 * selected.equity_need
            + 0.15 * selected.biodiversity_need
            + 0.15 * selected.feasibility
        )
        rationale = (
            f"Round {round_index}: selected {selected_cell_id} because {coalition_size}/{len(AGENT_NAMES)} "
            f"agents aligned on it and it preserved a shared reward of {shared_reward:.2f}."
        )

        steps.append(
            CoordinationStep(
                round_index=round_index,
                selected_cell_id=selected_cell_id,
                agent_preferences=preferences,
                coalition_size=coalition_size,
                coordination_score=round(coordination_score, 4),
                shared_reward=round(shared_reward, 4),
                rationale=rationale,
            )
        )
        selected_cells.append(selected_cell_id)
        remaining.remove(selected_cell_id)

    mean_coordination = mean(step.coordination_score for step in steps) if steps else 0.0
    conflict_rate = mean(1.0 - step.coordination_score for step in steps) if steps else 0.0
    final_shared_reward = sum(step.shared_reward for step in steps)
    selected_objects = [by_id[cell_id] for cell_id in selected_cells]
    indicator_summary = {
        "mean_cooling_gain": round(mean(cell.cooling_need for cell in selected_objects), 4) if selected_objects else 0.0,
        "mean_pollution_gain": round(mean(cell.pollution_need for cell in selected_objects), 4) if selected_objects else 0.0,
        "mean_equity_gain": round(mean(cell.equity_need for cell in selected_objects), 4) if selected_objects else 0.0,
        "mean_biodiversity_gain": round(mean(cell.biodiversity_need for cell in selected_objects), 4) if selected_objects else 0.0,
    }
    policy_summary = (
        f"The coordination episode allocated {len(selected_cells)} cells under budget {budget}. "
        f"Mean coordination was {mean_coordination:.2f}, conflict rate was {conflict_rate:.2f}, "
        f"and cumulative shared reward was {final_shared_reward:.2f}. "
        f"Selected cells targeted pollution={indicator_summary['mean_pollution_gain']:.2f}, cooling={indicator_summary['mean_cooling_gain']:.2f}, "
        f"equity={indicator_summary['mean_equity_gain']:.2f}, biodiversity={indicator_summary['mean_biodiversity_gain']:.2f}."
    )

    return CoordinationEpisode(
        budget=budget,
        selected_cells=selected_cells,
        steps=steps,
        final_shared_reward=round(final_shared_reward, 4),
        mean_coordination_score=round(mean_coordination, 4),
        conflict_rate=round(conflict_rate, 4),
        policy_summary=policy_summary,
        indicator_summary=indicator_summary,
    )


def align_ranked_cells_with_episode(ranked_cells: list[RankedCell], episode: CoordinationEpisode) -> list[RankedCell]:
    """Promote cells selected during the coordination episode for showcase clarity."""

    selected_order = {cell_id: index for index, cell_id in enumerate(episode.selected_cells)}
    return sorted(
        ranked_cells,
        key=lambda item: (
            item.cell.cell_id not in selected_order,
            selected_order.get(item.cell.cell_id, 999),
            -item.final_score,
        ),
    )


__all__ = ["AGENT_NAMES", "run_coordination_episode", "align_ranked_cells_with_episode"]
