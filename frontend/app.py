from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("EMERGENT_CANOPY_DATA_DIR", str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from urban_forest_ai.data_loader import (
    REQUIRED_DATA_FILES,
    available_area_names,
    ensure_data_dir,
    list_lads,
    list_msoas,
    load_parks,
    load_trees,
    missing_data_files,
    select_area,
)
from urban_forest_ai.pipeline import run_pipeline
from urban_forest_ai.settings import get_model
from urban_forest_ai.visualizations import (
    AGENT_DISPLAY,
    PALETTE,
    plot_coordination_timeline,
    plot_impact_summary,
    plot_radar_chart,
    plot_site_map,
    plot_vote_heatmap,
)


st.set_page_config(page_title="Emergent Canopy", layout="wide")


def init_state() -> None:
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None
    if "aoi_polygon" not in st.session_state:
        st.session_state["aoi_polygon"] = None
    if "trees" not in st.session_state:
        st.session_state["trees"] = None
    if "parks" not in st.session_state:
        st.session_state["parks"] = None
    if "cell_size" not in st.session_state:
        st.session_state["cell_size"] = 250.0
    if "autoplay_live_session" not in st.session_state:
        st.session_state["autoplay_live_session"] = False


_AGENT_COLORS = {
    "HeatAgent":         PALETTE["cooling"],
    "PollutionAgent":    PALETTE["pollution"],
    "EquityAgent":       PALETTE["equity"],
    "BiodiversityAgent": PALETTE["biodiversity"],
    "FeasibilityAgent":  PALETTE["feasibility"],
    "CritiqueAgent":     "#607D8B",
    "Judge":             "#37474F",
}

_STANCE_COLORS = {"support": "#4CAF50", "abstain": "#FF9800", "oppose": "#F44336"}


def _agent_chip(agent: str) -> str:
    color = _AGENT_COLORS.get(agent, "#9E9E9E")
    label = AGENT_DISPLAY.get(agent, agent)
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.8em;font-weight:600">{label}</span>'


def _stance_chip(stance: str) -> str:
    color = _STANCE_COLORS.get(stance, "#9E9E9E")
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.8em;font-weight:600">{stance}</span>'


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def save_uploaded_files(uploaded_files) -> list[str]:
    data_dir = ensure_data_dir()
    saved = []
    for uploaded in uploaded_files:
        target = data_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved.append(uploaded.name)
    available_area_names.cache_clear()
    return saved


@st.cache_data(show_spinner="Loading available areas...")
def _get_msoas() -> list[str]:
    return list_msoas()


@st.cache_data(show_spinner="Loading available areas...")
def _get_lads() -> list[str]:
    return list_lads()


@st.cache_data(show_spinner="Loading trees and parks...")
def _load_spatial_layers(area: str, level: str):
    _, aoi = select_area(area, level=level)
    return load_trees(aoi), load_parks(aoi)


def render_data_room() -> tuple[str, bool]:
    st.subheader("Data Room")
    st.write("Upload or refresh the local parquet files needed for a planning session.")

    data_dir = ensure_data_dir()
    missing = missing_data_files()
    present_rows = []
    for name in REQUIRED_DATA_FILES:
        path = data_dir / name
        present_rows.append(
            {
                "file": name,
                "status": "ready" if path.exists() else "missing",
                "size": format_bytes(path.stat().st_size) if path.exists() else "-",
            }
        )

    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(pd.DataFrame(present_rows), use_container_width=True, hide_index=True)
    with right:
        if missing:
            st.error(f"{len(missing)} required data files are missing.")
            st.code("\n".join(missing), language="text")
        else:
            st.success("All required data files are present.")

        uploaded_files = st.file_uploader(
            "Add parquet files",
            type=["parquet"],
            accept_multiple_files=True,
            help="Upload one or more of the required parquet files into the local data folder.",
        )
        if st.button("Save uploaded files", use_container_width=True, disabled=not uploaded_files):
            saved = save_uploaded_files(uploaded_files)
            st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")
            st.rerun()

    return str(data_dir.resolve()), len(missing) == 0


def render_controls(data_ready: bool) -> tuple[str, str, int, int, int, bool]:
    with st.sidebar:
        st.header("Planning Controls")

        level = st.radio("Area level", options=["MSOA", "LAD"], horizontal=True)
        if data_ready:
            if level == "MSOA":
                options = _get_msoas()
                default_idx = next((i for i, a in enumerate(options) if "Westminster" in a), 0)
            else:
                options = _get_lads()
                default_idx = next((i for i, a in enumerate(options) if a == "Westminster"), 0)
            area = st.selectbox("Area", options=options, index=default_idx)
        else:
            area = st.text_input("Area query", value="Westminster")

        cell_size = st.slider("Cell size (meters)", min_value=100, max_value=500, value=250, step=50)
        top_k = st.slider("Top recommendations", min_value=3, max_value=10, value=5)
        budget = st.slider("Planting budget (rounds)", min_value=1, max_value=6, value=3)

        debate_model = get_model("OPENAI_DEBATE_MODEL", "gpt-4o-mini")
        explainer_model = get_model("OPENAI_EXPLAINER_MODEL", "gpt-4o-mini")
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        if not has_key:
            try:
                has_key = "OPENAI_API_KEY" in st.secrets
            except Exception:
                pass
        if has_key:
            st.success(f"LLM Debate Active\nModels: `{debate_model}` / `{explainer_model}`")
        else:
            st.warning("Fallback Mode (deterministic)\nSet OPENAI_API_KEY to enable live LLM debate.")

        run = st.button(
            "Run analysis",
            type="primary",
            use_container_width=True,
            disabled=not data_ready,
        )
    return area, level.lower(), cell_size, top_k, budget, run


def render_overview(result) -> None:
    aoi_polygon = st.session_state["aoi_polygon"]
    trees = st.session_state["trees"]
    parks = st.session_state["parks"]
    cell_size = st.session_state["cell_size"]

    st.subheader(result.area_name)
    st.metric("Candidate cells analysed", result.candidates_considered)

    st.subheader("Where Should We Plant?")
    st.caption(
        "Each dot is a candidate location. Starred sites are the final recommendations — "
        "darker green means a higher overall score. The dark outline shows the selected area boundary."
    )
    st.plotly_chart(
        plot_site_map(result, aoi_polygon=aoi_polygon, trees=trees, parks=parks, cell_size=cell_size),
        use_container_width=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("How Each Site Scores")
        st.caption(
            "The larger and more balanced the shape, the better a site performs across all goals. "
            "Solid lines are the recommended sites."
        )
        st.plotly_chart(plot_radar_chart(result), use_container_width=True)
    with col_right:
        st.subheader("How Advisers Voted")
        st.caption(
            "Green = the adviser supports this site. Amber = neutral. Red = the adviser opposes it. "
            "Numbers show the adviser's confidence score."
        )
        st.plotly_chart(plot_vote_heatmap(result), use_container_width=True)

    if result.coordination_episode:
        st.subheader("The Deliberation Process")
        st.write(result.coordination_episode.policy_summary)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(
                "Each round, the advisers collectively choose the next best planting site. "
                "The blue bar shows how much benefit the choice delivers; green shows how many advisers agreed."
            )
            st.plotly_chart(plot_coordination_timeline(result), use_container_width=True)
        with col_b:
            st.caption(
                "How the recommended sites collectively perform against four urban climate goals. "
                "Higher is better; the dashed line marks the midpoint."
            )
            st.plotly_chart(plot_impact_summary(result), use_container_width=True)


def render_live_session(result) -> None:
    st.subheader("Live Agents Debate and Planning Session")
    replay = st.button("Replay live session", use_container_width=False)
    should_play = st.session_state["autoplay_live_session"] or replay

    if not should_play:
        st.info("Click replay to watch the coordination and debate unfold.")
        return

    st.session_state["autoplay_live_session"] = False
    timeline = st.empty()
    progress = st.progress(0)
    session_log = st.container()

    if result.coordination_episode:
        total_steps = len(result.coordination_episode.steps) + len(result.ranked_cells)
        current_index = 0
        for step in result.coordination_episode.steps:
            current_index += 1
            progress.progress(current_index / total_steps)
            timeline.info(
                f"Round {step.round_index}: coalition selected `{step.selected_cell_id}` "
                f"with coordination score {step.coordination_score:.2f} and shared reward {step.shared_reward:.2f}."
            )
            with session_log:
                st.markdown(f"**Coordination round {step.round_index}**")
                st.write(step.rationale)
                st.json(step.agent_preferences, expanded=False)
            time.sleep(0.4)

        for item in result.ranked_cells:
            current_index += 1
            progress.progress(current_index / total_steps)
            timeline.success(
                f"Debate resolved for `{item.cell.cell_id}` with final score {item.final_score:.2f}."
            )
            with session_log:
                st.markdown(f"**Debate resolution for {item.cell.cell_id}**")
                st.write(item.summary)
                for turn in item.transcript:
                    chip = _agent_chip(turn.agent)
                    phase_label = f"`{turn.phase}`"
                    st.markdown(f"{chip} {phase_label}  \n{turn.content}", unsafe_allow_html=True)
                st.markdown("**Final votes**")
                for vote in item.votes:
                    chip = _agent_chip(vote.agent)
                    stance = _stance_chip(vote.stance)
                    st.markdown(
                        f"{chip} {stance} &nbsp; score `{vote.score:.2f}`",
                        unsafe_allow_html=True,
                    )
            time.sleep(0.3)


def render_results(result) -> None:
    with st.expander("Technical details", expanded=False):
        st.subheader("Baseline comparison")
        baseline_df = pd.DataFrame(
            [{"Metric": k, "Score": v} for k, v in result.baseline_summary.items()]
        )
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)

        if result.coordination_episode:
            st.subheader("Coordination steps")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "round": step.round_index,
                            "selected_cell": step.selected_cell_id,
                            "coalition_size": step.coalition_size,
                            "coordination_score": step.coordination_score,
                            "shared_reward": step.shared_reward,
                        }
                        for step in result.coordination_episode.steps
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Raw ranked cells")
        rows = []
        for item in result.ranked_cells:
            rows.append({
                "cell_id":         item.cell.cell_id,
                "debate_score":    item.final_score,
                "feature_score":   item.cell.combined_score,
                "cooling":         item.cell.cooling_need,
                "pollution":       item.cell.pollution_need,
                "equity":          item.cell.equity_need,
                "biodiversity":    item.cell.biodiversity_need,
                "feasibility":     item.cell.feasibility,
                "budget_fit":      item.cell.budget_fit,
                "debate_turns":    len(item.transcript),
                "buildings":       item.cell.evidence["nearby_buildings"],
                "trees":           item.cell.evidence["nearby_trees"],
                "roads":           item.cell.evidence["nearby_road_nodes"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.code(result.trace_path, language="text")
        st.json(result.denario_status, expanded=False)

    with st.expander("Debate transcript", expanded=False):
        for item in result.ranked_cells:
            with st.expander(f"{item.cell.cell_id} · debate score {item.final_score:.2f}", expanded=False):
                st.write(item.summary)
                if item.llm_explanation:
                    st.markdown(f"**LLM explanation**  \n{item.llm_explanation}")
                st.markdown("**Shared transcript**")
                for turn in item.transcript:
                    chip = _agent_chip(turn.agent)
                    st.markdown(
                        f"{chip} &nbsp; `{turn.phase}`  \n{turn.content}",
                        unsafe_allow_html=True,
                    )
                st.markdown("**Reason chain**")
                for reason in item.reason_chain:
                    st.write(f"- {reason}")
                st.markdown("**Final votes**")
                for vote in item.votes:
                    chip = _agent_chip(vote.agent)
                    stance = _stance_chip(vote.stance)
                    st.markdown(
                        f"{chip} {stance} &nbsp; score `{vote.score:.2f}`  \n{vote.rationale}",
                        unsafe_allow_html=True,
                    )


init_state()

st.title("Emergent Canopy")
st.caption("A multi-agent urban tree planting planner — helping cities grow greener, fairer, and cooler.")

data_path, data_ready = render_data_room()
area, level, cell_size, top_k, budget, run = render_controls(data_ready)
st.caption(f"Local data directory: `{data_path}`")

if run:
    with st.status("Running analysis...", expanded=True) as _status:
        _stage_text = st.empty()
        _debate_bar = st.progress(0, text="Waiting for debate...")

        def _on_stage(msg: str) -> None:
            _stage_text.write(msg)

        def _on_cell_done(done: int, total: int, cell_id: str) -> None:
            _debate_bar.progress(done / total, text=f"Debating cell {done}/{total}: `{cell_id}`")

        _, aoi_polygon = select_area(area, level=level)
        trees, parks = _load_spatial_layers(area, level=level)
        result = run_pipeline(
            area_query=area,
            level=level,
            cell_size=float(cell_size),
            top_k=int(top_k),
            budget=int(budget),
            on_stage=_on_stage,
            on_cell_done=_on_cell_done,
        )
        _status.update(label="Analysis complete!", state="complete", expanded=False)
        st.session_state["latest_result"] = result
        st.session_state["aoi_polygon"] = aoi_polygon
        st.session_state["trees"] = trees
        st.session_state["parks"] = parks
        st.session_state["cell_size"] = float(cell_size)
        st.session_state["autoplay_live_session"] = True

result = st.session_state["latest_result"]

if result is not None:
    render_overview(result)
    render_live_session(result)
    render_results(result)
else:
    st.info("Upload data if needed, choose an area, and click **Run analysis**.")
