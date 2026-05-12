from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import plotly.io as pio


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
os.environ.setdefault("EMERGENT_CANOPY_DATA_DIR", str(REPO_ROOT / "data"))

from urban_forest_ai.data_loader import load_parks, load_trees, select_area
from urban_forest_ai.models import (  # noqa: E402
    AgentVote,
    CellScore,
    CoordinationEpisode,
    CoordinationStep,
    DebateTurn,
    PipelineResult,
    RankedCell,
)
from urban_forest_ai.visualizations import (  # noqa: E402
    plot_coordination_timeline,
    plot_impact_summary,
    plot_radar_chart,
    plot_site_map,
    plot_vote_heatmap,
)


def _latest_trace() -> Path:
    traces = sorted((REPO_ROOT / "outputs").glob("*_trace.json"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise FileNotFoundError("No trace JSON files found in outputs/. Run the analysis once first.")
    return traces[-1]


def _cell_score(payload: dict[str, Any]) -> CellScore:
    return CellScore(**payload)


def _vote(payload: dict[str, Any]) -> AgentVote:
    return AgentVote(**payload)


def _turn(payload: dict[str, Any]) -> DebateTurn:
    return DebateTurn(**payload)


def _coordination_episode(payload: dict[str, Any] | None) -> CoordinationEpisode | None:
    if not payload:
        return None
    steps = [CoordinationStep(**step) for step in payload.get("steps", [])]
    return CoordinationEpisode(
        budget=payload["budget"],
        selected_cells=payload["selected_cells"],
        steps=steps,
        final_shared_reward=payload["final_shared_reward"],
        mean_coordination_score=payload["mean_coordination_score"],
        conflict_rate=payload["conflict_rate"],
        policy_summary=payload["policy_summary"],
        indicator_summary=payload.get("indicator_summary", {}),
    )


def _result_from_trace(trace: Path) -> PipelineResult:
    payload = json.loads(trace.read_text())
    ranked_cells = []
    for item in payload["ranked_cells"]:
        ranked_cells.append(
            RankedCell(
                cell=_cell_score(item["cell"]),
                votes=[_vote(vote) for vote in item.get("votes", [])],
                transcript=[_turn(turn) for turn in item.get("transcript", [])],
                final_score=item["final_score"],
                summary=item["summary"],
                reason_chain=item.get("reason_chain", []),
                llm_explanation=item.get("llm_explanation"),
            )
        )
    return PipelineResult(
        area_name=payload["area_name"],
        denario_status={},
        candidates_considered=len(ranked_cells),
        ranked_cells=ranked_cells,
        baseline_summary=payload.get("baseline_summary", {}),
        coordination_episode=_coordination_episode(payload.get("coordination_episode")),
        trace_path=str(trace.resolve()),
    )


def _figure_html(fig, include_plotlyjs: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )


def _screenshot_data_uri() -> str:
    screenshot = REPO_ROOT / "outputs" / "streamlit-page.png"
    if not screenshot.exists():
        return ""
    encoded = base64.b64encode(screenshot.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _safe(text: Any) -> str:
    return html.escape(str(text))


def _metric_cards(result: PipelineResult) -> str:
    episode = result.coordination_episode
    cards = [
        ("Area", result.area_name),
        ("Saved recommendations", len(result.ranked_cells)),
    ]
    if episode:
        cards.extend(
            [
                ("Budget rounds", episode.budget),
                ("Coordination", f"{episode.mean_coordination_score:.2f}"),
                ("Conflict", f"{episode.conflict_rate:.2f}"),
                ("Shared reward", f"{episode.final_shared_reward:.2f}"),
            ]
        )
    return "\n".join(
        f"<div class='metric'><span>{_safe(label)}</span><strong>{_safe(value)}</strong></div>"
        for label, value in cards
    )


def _recommendation_table(result: PipelineResult) -> str:
    rows = []
    for rank, item in enumerate(sorted(result.ranked_cells, key=lambda x: x.final_score, reverse=True), 1):
        cell = item.cell
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{_safe(cell.cell_id)}</td>"
            f"<td>{item.final_score:.2f}</td>"
            f"<td>{cell.combined_score:.2f}</td>"
            f"<td>{cell.cooling_need:.2f}</td>"
            f"<td>{cell.pollution_need:.2f}</td>"
            f"<td>{cell.equity_need:.2f}</td>"
            f"<td>{cell.biodiversity_need:.2f}</td>"
            f"<td>{cell.feasibility:.2f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Rank</th><th>Cell</th><th>Debate</th><th>Feature</th>"
        "<th>Cooling</th><th>Air</th><th>Equity</th><th>Nature</th><th>Feasible</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _debate_sections(result: PipelineResult) -> str:
    sections = []
    for item in sorted(result.ranked_cells, key=lambda x: x.final_score, reverse=True):
        vote_rows = "\n".join(
            "<tr>"
            f"<td>{_safe(vote.agent)}</td>"
            f"<td><span class='stance {_safe(vote.stance)}'>{_safe(vote.stance)}</span></td>"
            f"<td>{vote.score:.2f}</td>"
            f"<td>{_safe(vote.rationale)}</td>"
            "</tr>"
            for vote in item.votes
        )
        turns = "\n".join(
            "<div class='turn'>"
            f"<div><strong>{_safe(turn.agent)}</strong> <span>{_safe(turn.phase)}</span></div>"
            f"<p>{_safe(turn.content)}</p>"
            "</div>"
            for turn in item.transcript
        )
        reasons = "\n".join(f"<li>{_safe(reason)}</li>" for reason in item.reason_chain)
        explanation = (
            f"<div class='llm'><h4>LLM Explanation</h4><p>{_safe(item.llm_explanation)}</p></div>"
            if item.llm_explanation
            else ""
        )
        sections.append(
            "<section class='cell-section'>"
            f"<h3>{_safe(item.cell.cell_id)} · debate score {item.final_score:.2f}</h3>"
            f"<p class='summary'>{_safe(item.summary)}</p>"
            f"{explanation}"
            "<h4>Final Votes</h4>"
            "<table><thead><tr><th>Agent</th><th>Stance</th><th>Score</th><th>Rationale</th></tr></thead>"
            f"<tbody>{vote_rows}</tbody></table>"
            "<h4>Reason Chain</h4>"
            f"<ul>{reasons}</ul>"
            "<h4>Saved Debate Transcript</h4>"
            f"{turns}"
            "</section>"
        )
    return "\n".join(sections)


def _coordination_section(result: PipelineResult) -> str:
    episode = result.coordination_episode
    if not episode:
        return "<p>No coordination episode was saved in this trace.</p>"
    rows = "\n".join(
        "<tr>"
        f"<td>{step.round_index}</td>"
        f"<td>{_safe(step.selected_cell_id)}</td>"
        f"<td>{step.coalition_size}</td>"
        f"<td>{step.coordination_score:.2f}</td>"
        f"<td>{step.shared_reward:.2f}</td>"
        f"<td>{_safe(step.rationale)}</td>"
        "</tr>"
        for step in episode.steps
    )
    return (
        f"<p>{_safe(episode.policy_summary)}</p>"
        "<table><thead><tr><th>Round</th><th>Selected Cell</th><th>Coalition</th>"
        "<th>Coordination</th><th>Reward</th><th>Rationale</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_report(result: PipelineResult, trace: Path, out: Path) -> None:
    aoi_polygon = trees = parks = None
    try:
        _, aoi_polygon = select_area(result.area_name, level="msoa")
        trees = load_trees(aoi_polygon)
        parks = load_parks(aoi_polygon)
    except Exception:
        pass

    chart_html = [
        _figure_html(plot_site_map(result, aoi_polygon=aoi_polygon, trees=trees, parks=parks, cell_size=150), True),
        _figure_html(plot_radar_chart(result)),
        _figure_html(plot_vote_heatmap(result)),
        _figure_html(plot_coordination_timeline(result)),
        _figure_html(plot_impact_summary(result)),
    ]
    screenshot_uri = _screenshot_data_uri()
    screenshot_html = (
        f"<img class='screenshot' src='{screenshot_uri}' alt='Saved Streamlit page screenshot'>"
        if screenshot_uri
        else ""
    )
    source_json = _safe(json.dumps(json.loads(trace.read_text()), indent=2))
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Emergent Canopy Saved Analysis · {html.escape(result.area_name)}</title>
  <style>
    body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #102033; background: #f6f8fb; }}
    header {{ padding: 48px 7vw 32px; color: white; background: linear-gradient(135deg, #12372a, #1f6b52); }}
    header h1 {{ margin: 0 0 10px; font-size: 42px; }}
    header p {{ max-width: 920px; font-size: 18px; line-height: 1.5; }}
    main {{ padding: 32px 7vw 72px; }}
    section {{ background: white; border: 1px solid #dbe4ee; border-radius: 14px; padding: 24px; margin: 0 0 24px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06); }}
    h2 {{ margin-top: 0; font-size: 26px; }}
    h3 {{ margin-top: 0; font-size: 22px; }}
    h4 {{ margin-bottom: 10px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-top: 24px; }}
    .metric {{ background: #eef7f2; border: 1px solid #cce8d7; border-radius: 12px; padding: 16px; }}
    .metric span {{ display: block; color: #526373; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
    .metric strong {{ display: block; margin-top: 8px; font-size: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 20px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5edf5; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f6fa; font-weight: 700; }}
    .plot {{ margin: 18px 0; }}
    .cell-section {{ border-left: 5px solid #2f855a; }}
    .summary {{ font-size: 16px; line-height: 1.55; }}
    .turn {{ border-left: 3px solid #cbd5e1; padding: 10px 14px; margin: 10px 0; background: #f8fafc; border-radius: 8px; }}
    .turn span {{ color: #64748b; font-size: 13px; margin-left: 8px; }}
    .turn p {{ margin: 6px 0 0; line-height: 1.5; }}
    .stance {{ color: white; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 700; }}
    .support {{ background: #2f855a; }}
    .abstain {{ background: #b7791f; }}
    .oppose {{ background: #c53030; }}
    .llm {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 12px; padding: 14px; margin: 12px 0; }}
    .screenshot {{ width: 100%; border: 1px solid #dbe4ee; border-radius: 12px; }}
    details {{ margin-top: 12px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #0f172a; color: #e2e8f0; padding: 18px; border-radius: 12px; max-height: 520px; overflow: auto; }}
    .note {{ color: #526373; }}
  </style>
</head>
<body>
  <header>
    <h1>Emergent Canopy Saved Analysis</h1>
    <p>This is a static export of a completed Streamlit analysis. It preserves the recommendations, charts, votes, debate transcript, explanations, and raw trace so the OpenAI analysis does not need to be run again.</p>
    <div class="metrics">{_metric_cards(result)}</div>
  </header>
  <main>
    <section>
      <h2>How To Use This Saved Page</h2>
      <p>This HTML file can be opened directly in a browser. It is based on <code>{_safe(trace.name)}</code> and does not call OpenAI or rerun the planning pipeline. The map may still request background map tiles, but the saved recommendations and transcript are embedded in this file.</p>
    </section>
    <section>
      <h2>Visual Snapshot</h2>
      <p class="note">A saved screenshot of the original Streamlit page is embedded below as a visual fallback.</p>
      {screenshot_html}
    </section>
    <section>
      <h2>Where Should We Plant?</h2>
      <div class="plot">{chart_html[0]}</div>
    </section>
    <section>
      <h2>Recommendation Scores</h2>
      {_recommendation_table(result)}
      <div class="plot">{chart_html[1]}</div>
    </section>
    <section>
      <h2>How Advisers Voted</h2>
      <div class="plot">{chart_html[2]}</div>
    </section>
    <section>
      <h2>The Deliberation Process</h2>
      {_coordination_section(result)}
      <div class="plot">{chart_html[3]}</div>
      <div class="plot">{chart_html[4]}</div>
    </section>
    <section>
      <h2>Saved Agent Debate and Explanations</h2>
      {_debate_sections(result)}
    </section>
    <section>
      <h2>Raw Audit Trace</h2>
      <details>
        <summary>Show embedded JSON trace</summary>
        <pre>{source_json}</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "emergent_canopy_saved_analysis.html")
    args = parser.parse_args()
    trace = args.trace or _latest_trace()
    result = _result_from_trace(trace)
    _build_report(result, trace, args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
