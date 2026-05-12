from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from urban_forest_ai.models import PipelineResult

# --- Constants ---

PALETTE = {
    "cooling":      "#2196F3",
    "pollution":    "#FF9800",
    "equity":       "#9C27B0",
    "biodiversity": "#4CAF50",
    "feasibility":  "#795548",
}

AGENT_ORDER = [
    "HeatAgent",
    "PollutionAgent",
    "EquityAgent",
    "BiodiversityAgent",
    "FeasibilityAgent",
]

AGENT_DISPLAY = {
    "HeatAgent":         "Heat / Cooling",
    "PollutionAgent":    "Air Quality",
    "EquityAgent":       "Community Equity",
    "BiodiversityAgent": "Nature & Biodiversity",
    "FeasibilityAgent":  "Ease of Planting",
}

DIMENSION_KEYS   = ["cooling_need", "pollution_need", "equity_need", "biodiversity_need", "feasibility"]
DIMENSION_LABELS = ["Cooling Priority", "Air Quality Priority", "Community Equity", "Nature Value", "Ease of Planting"]

STANCE_TO_NUM = {"support": 1, "abstain": 0, "oppose": -1}

STANCE_COLORSCALE = [
    [0.0, "#F44336"],
    [0.5, "#FFC107"],
    [1.0, "#4CAF50"],
]

# Colours cycled across ranked cells in the radar chart
_RADAR_COLORS = [
    "#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#795548",
    "#E91E63", "#00BCD4", "#FF5722", "#607D8B", "#8BC34A",
]

# --- Helpers ---

_transformer = None


def _bng_to_latlon(x: float, y: float) -> tuple[float, float]:
    """Convert EPSG:27700 (British National Grid) to (lat, lon) WGS84."""
    global _transformer
    if _transformer is None:
        from pyproj import Transformer
        _transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = _transformer.transform(x, y)
    return lat, lon


def _empty_figure(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=msg, x=0.5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --- Plot functions ---

def _cell_rect_latlon(cx: float, cy: float, cell_size: float):
    """Return (lats, lons) for a closed BNG square centred at (cx, cy)."""
    h = cell_size / 2
    corners = [
        (cx - h, cy - h),
        (cx + h, cy - h),
        (cx + h, cy + h),
        (cx - h, cy + h),
        (cx - h, cy - h),  # close the ring
    ]
    lats = [_bng_to_latlon(x, y)[0] for x, y in corners]
    lons = [_bng_to_latlon(x, y)[1] for x, y in corners]
    return lats, lons


def _polygon_to_latlon_trace(polygon) -> tuple[list[float], list[float]]:
    """Convert a shapely Polygon or MultiPolygon (EPSG:27700) to (lats, lons) for a line trace.
    Rings are separated by None to lift the pen between disconnected parts."""
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(polygon, Polygon):
        parts = [polygon]
    elif isinstance(polygon, MultiPolygon):
        parts = list(polygon.geoms)
    else:
        return [], []

    lats: list[float | None] = []
    lons: list[float | None] = []
    for part in parts:
        coords = list(part.exterior.coords)
        for x, y in coords:
            lat, lon = _bng_to_latlon(x, y)
            lats.append(lat)
            lons.append(lon)
        lats.append(None)
        lons.append(None)
    return lats, lons


def plot_site_map(
    result: "PipelineResult",
    aoi_polygon=None,
    trees=None,
    parks=None,
    cell_size: float = 250.0,
    max_trees: int = 3000,
) -> go.Figure:
    """Map of recommended planting sites with a B&W basemap.

    Parameters
    ----------
    result:
        Pipeline output containing ranked cells.
    aoi_polygon:
        Optional shapely geometry (EPSG:27700) for the selected MSOA boundary.
    trees:
        Optional numpy array of shape (N, 2) with BNG (x, y) tree point coordinates.
    parks:
        Optional list of shapely geometries (EPSG:27700) for park polygons.
    cell_size:
        Side length of each grid cell in metres, used to draw cell footprints.
    max_trees:
        Maximum number of tree points to render (randomly subsampled if exceeded).
    """
    if not result.ranked_cells:
        return _empty_figure("No candidate sites to display")

    selected_ids: set[str] = set()
    if result.coordination_episode:
        selected_ids = set(result.coordination_episode.selected_cells)

    lats_c, lons_c, sizes_c, scores_c, custom_c = [], [], [], [], []
    lats_s, lons_s, sizes_s, scores_s, custom_s, texts_s = [], [], [], [], [], []
    rank_counter = {item.cell.cell_id: i + 1 for i, item in
                    enumerate(sorted(result.ranked_cells, key=lambda x: x.final_score, reverse=True))}

    for item in result.ranked_cells:
        cx, cy = item.cell.centroid_x, item.cell.centroid_y
        if cx == 0 and cy == 0:
            continue
        lat, lon = _bng_to_latlon(cx, cy)
        rank = rank_counter[item.cell.cell_id]
        size = 10 + 20 * item.cell.combined_score
        cd = [
            item.cell.cell_id,
            round(item.final_score, 2),
            round(item.cell.cooling_need, 2),
            round(item.cell.pollution_need, 2),
            round(item.cell.equity_need, 2),
            rank,
        ]
        if item.cell.cell_id in selected_ids:
            lats_s.append(lat); lons_s.append(lon)
            sizes_s.append(size * 1.5); scores_s.append(item.final_score)
            custom_s.append(cd); texts_s.append(f"#{rank}")
        else:
            lats_c.append(lat); lons_c.append(lon)
            sizes_c.append(size); scores_c.append(item.final_score)
            custom_c.append(cd)

    hover_tmpl = (
        "<b>%{customdata[0]}</b><br>"
        "Debate score: %{customdata[1]}<br>"
        "Cooling: %{customdata[2]} · Air quality: %{customdata[3]} · Equity: %{customdata[4]}<br>"
        "Rank: #%{customdata[5]}"
        "<extra></extra>"
    )
    green_scale = [[0, "#a8d5a2"], [1, "#1a6e1a"]]

    fig = go.Figure()

    if lats_c:
        fig.add_trace(go.Scattermapbox(
            lat=lats_c, lon=lons_c,
            mode="markers",
            marker=dict(
                size=sizes_c,
                color=scores_c,
                colorscale=green_scale,
                cmin=0, cmax=1,
                opacity=0.8,
            ),
            customdata=custom_c,
            hovertemplate=hover_tmpl,
            name="Candidate sites",
        ))

    if lats_s:
        fig.add_trace(go.Scattermapbox(
            lat=lats_s, lon=lons_s,
            mode="markers+text",
            marker=dict(
                size=sizes_s,
                color=scores_s,
                colorscale=green_scale,
                cmin=0, cmax=1,
                opacity=1.0,
            ),
            text=texts_s,
            textposition="top right",
            customdata=custom_s,
            hovertemplate="<b>%{customdata[0]}</b> RECOMMENDED<br>"
                          "Debate score: %{customdata[1]}<br>"
                          "Cooling: %{customdata[2]} · Air quality: %{customdata[3]} · Equity: %{customdata[4]}<br>"
                          "Rank: #%{customdata[5]}"
                          "<extra></extra>",
            name="Recommended sites",
        ))

    # --- Parks (filled polygons, drawn first so they sit behind everything) ---
    if parks:
        park_lats: list = []
        park_lons: list = []
        for park_geom in parks:
            p_lats, p_lons = _polygon_to_latlon_trace(park_geom)
            park_lats.extend(p_lats)
            park_lons.extend(p_lons)
        fig.add_trace(go.Scattermapbox(
            lat=park_lats,
            lon=park_lons,
            mode="lines",
            fill="toself",
            fillcolor="rgba(76, 175, 80, 0.15)",
            line=dict(color="rgba(56, 142, 60, 0.35)", width=1),
            hoverinfo="skip",
            name="Existing parks",
            showlegend=True,
        ))

    # --- Trees (subsampled point cloud) ---
    if trees is not None and len(trees) > 0:
        import numpy as np
        pts = trees
        if len(pts) > max_trees:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pts), size=max_trees, replace=False)
            pts = pts[idx]
        t_lats, t_lons = [], []
        for x, y in pts:
            lat, lon = _bng_to_latlon(float(x), float(y))
            t_lats.append(lat)
            t_lons.append(lon)
        n_shown = len(t_lats)
        n_total = len(trees)
        tree_label = "Existing trees" + (f" (sample of {n_shown:,})" if n_total > max_trees else "")
        fig.add_trace(go.Scattermapbox(
            lat=t_lats,
            lon=t_lons,
            mode="markers",
            marker=dict(size=4, color="rgba(46, 125, 50, 0.30)"),
            hoverinfo="skip",
            name=tree_label,
            showlegend=True,
        ))

    # --- MSOA boundary (drawn on top of parks/trees, below site markers) ---
    if aoi_polygon is not None:
        b_lats, b_lons = _polygon_to_latlon_trace(aoi_polygon)
        fig.add_trace(go.Scattermapbox(
            lat=b_lats,
            lon=b_lons,
            mode="lines",
            line=dict(color="#333333", width=2),
            hoverinfo="skip",
            name="Area boundary",
            showlegend=True,
        ))

    all_lats = lats_c + lats_s
    all_lons = lons_c + lons_s
    center_lat = sum(all_lats) / len(all_lats) if all_lats else 51.5
    center_lon = sum(all_lons) / len(all_lons) if all_lons else -0.12

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=14 if len(result.ranked_cells) > 1 else 15,
        mapbox_center=dict(lat=center_lat, lon=center_lon),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        title=dict(text=f"Recommended Planting Sites — {result.area_name}", x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_radar_chart(result: "PipelineResult") -> go.Figure:
    """Radar/spider chart comparing 5 scoring dimensions across all ranked cells."""
    if not result.ranked_cells:
        return _empty_figure("No candidate sites to display")

    selected_ids: set[str] = set()
    if result.coordination_episode:
        selected_ids = set(result.coordination_episode.selected_cells)

    cats_closed = DIMENSION_LABELS + [DIMENSION_LABELS[0]]
    fig = go.Figure()

    for i, item in enumerate(result.ranked_cells):
        cell = item.cell
        values = [
            cell.cooling_need,
            cell.pollution_need,
            cell.equity_need,
            cell.biodiversity_need,
            cell.feasibility,
        ]
        vals_closed = values + [values[0]]
        is_sel = cell.cell_id in selected_ids
        color = _RADAR_COLORS[i % len(_RADAR_COLORS)]
        label = f"{cell.cell_id} (score {item.final_score:.2f})" + (" ★" if is_sel else "")

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.10),
            line=dict(color=color, width=3 if is_sel else 1.5, dash="solid" if is_sel else "dot"),
            name=label,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=True,
        legend=dict(font=dict(size=10)),
        title=dict(text="Site Comparison: What Each Location Is Good For", x=0.5),
        height=500,
        margin=dict(t=60),
    )
    return fig


def plot_vote_heatmap(result: "PipelineResult") -> go.Figure:
    """Heatmap of agent stances (support/abstain/oppose) across ranked cells."""
    if not result.ranked_cells:
        return _empty_figure("No candidate sites to display")

    sorted_cells = sorted(result.ranked_cells, key=lambda x: x.final_score, reverse=True)
    col_labels = [f"{item.cell.cell_id}<br>score {item.final_score:.2f}" for item in sorted_cells]

    z_matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []

    for agent in AGENT_ORDER:
        row_z, row_text = [], []
        for item in sorted_cells:
            vote = next((v for v in item.votes if v.agent == agent), None)
            if vote:
                row_z.append(float(STANCE_TO_NUM[vote.stance]))
                row_text.append(f"{vote.score:.2f}")
            else:
                row_z.append(0.0)
                row_text.append("—")
        z_matrix.append(row_z)
        text_matrix.append(row_text)

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=z_matrix,
        x=col_labels,
        y=[AGENT_DISPLAY[a] for a in AGENT_ORDER],
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=13, color="black"),
        colorscale=STANCE_COLORSCALE,
        zmin=-1, zmax=1,
        showscale=False,
        hovertemplate="Agent: %{y}<br>Site: %{x}<br>Score: %{text}<extra></extra>",
    ))

    for label, color, symbol in [
        ("Support", "#4CAF50", "square"),
        ("Abstain", "#FFC107", "square"),
        ("Oppose",  "#F44336", "square"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color, symbol=symbol),
            name=label,
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="How Each AI Adviser Voted on Each Site", x=0.5),
        xaxis=dict(title="Candidate Sites (ranked left to right)", tickfont=dict(size=11)),
        yaxis=dict(title="", tickfont=dict(size=12)),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=160, t=60),
    )
    return fig


def plot_coordination_timeline(result: "PipelineResult") -> go.Figure:
    """Grouped bar chart showing shared reward and adviser agreement per coordination round."""
    if not result.coordination_episode:
        return _empty_figure("No coordination episode to display")

    steps = result.coordination_episode.steps
    if not steps:
        return _empty_figure("No coordination rounds recorded")

    rounds      = [f"Round {s.round_index}" for s in steps]
    rewards     = [s.shared_reward for s in steps]
    coalitions  = [s.coalition_size / 5.0 for s in steps]
    cell_labels = [s.selected_cell_id for s in steps]
    coalition_abs = [[s.coalition_size] for s in steps]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=rounds,
        y=rewards,
        name="Shared Benefit Score",
        marker_color="#2196F3",
        text=cell_labels,
        textposition="inside",
        textfont=dict(color="white", size=12),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Selected: %{text}<br>"
            "Benefit score: %{y:.3f}"
            "<extra></extra>"
        ),
        yaxis="y",
    ))

    fig.add_trace(go.Bar(
        x=rounds,
        y=coalitions,
        name="Adviser Agreement",
        marker_color="#4CAF50",
        opacity=0.75,
        customdata=coalition_abs,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Advisers agreed: %{customdata[0]}/5<br>"
            "Agreement: %{y:.0%}"
            "<extra></extra>"
        ),
        yaxis="y2",
    ))

    max_reward = max(rewards) if rewards else 1.0
    fig.update_layout(
        barmode="group",
        yaxis=dict(
            title=dict(text="Shared Benefit Score", font=dict(color="#2196F3")),
            range=[0, max_reward * 1.25],
            tickfont=dict(color="#2196F3"),
        ),
        yaxis2=dict(
            title=dict(text="Adviser Agreement", font=dict(color="#4CAF50")),
            overlaying="y",
            side="right",
            range=[0, 1.05],
            tickformat=".0%",
            tickfont=dict(color="#4CAF50"),
        ),
        title=dict(text="How Advisers Reached Agreement Each Round", x=0.5),
        xaxis=dict(title="Planting Round"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(t=60),
    )
    return fig


def plot_impact_summary(result: "PipelineResult") -> go.Figure:
    """Horizontal bar chart showing the mean climate impact gains from selected sites."""
    if not result.coordination_episode:
        return _empty_figure("No coordination episode to display")

    indicator_config = [
        ("mean_cooling_gain",      "Heat Reduction in Selected Areas",  PALETTE["cooling"]),
        ("mean_pollution_gain",    "Air Quality Improvement",           PALETTE["pollution"]),
        ("mean_equity_gain",       "Benefit to Underserved Communities", PALETTE["equity"]),
        ("mean_biodiversity_gain", "Nature & Wildlife Enhancement",     PALETTE["biodiversity"]),
    ]

    summary = result.coordination_episode.indicator_summary
    labels, values, colors = [], [], []
    for key, label, color in indicator_config:
        labels.append(label)
        values.append(summary.get(key, 0.0))
        colors.append(color)

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
        textfont=dict(size=14),
        cliponaxis=False,
        hovertemplate="%{y}<br>Score: %{x:.2f}<extra></extra>",
    ))

    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="Moderate impact",
        annotation_position="top right",
        annotation_font_size=11,
    )

    if values:
        best_idx = values.index(max(values))
        fig.add_annotation(
            x=values[best_idx],
            y=labels[best_idx],
            text="Strongest outcome",
            showarrow=True,
            arrowhead=2,
            ax=55,
            ay=-20,
            font=dict(size=11),
        )

    fig.update_layout(
        title=dict(text="What This Planting Plan Achieves", x=0.5),
        xaxis=dict(title="Impact Score (0 = none, 1 = maximum)", range=[0, 1.25]),
        yaxis=dict(title=""),
        height=350,
        margin=dict(l=240, r=80, t=60),
    )
    return fig
