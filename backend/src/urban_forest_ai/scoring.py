from __future__ import annotations

from math import sqrt

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.strtree import STRtree

from .models import CellScore


def make_grid(aoi, cell_size: float = 250.0) -> list[object]:
    minx, miny, maxx, maxy = aoi.bounds
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + cell_size, y + cell_size)
            clipped = cell.intersection(aoi)
            if not clipped.is_empty and clipped.area > 0:
                cells.append(clipped)
            y += cell_size
        x += cell_size
    return cells


def _normalize(values: list[float], invert: bool = False) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        base = [0.5 for _ in values]
    else:
        base = [(value - lo) / (hi - lo) for value in values]
    if invert:
        return [1.0 - value for value in base]
    return base


def _count_points(points: np.ndarray, center: Point, radius: float) -> int:
    if points.size == 0:
        return 0
    deltas = points - np.array([[center.x, center.y]])
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    return int((dists <= radius).sum())


def score_cells(cells, buildings, trees: np.ndarray, parks, road_nodes: np.ndarray) -> list[CellScore]:
    building_tree = STRtree(buildings) if buildings else None
    park_tree = STRtree(parks) if parks else None
    road_points = road_nodes if road_nodes.size else np.empty((0, 2))

    raw_rows = []
    for idx, cell in enumerate(cells):
        centroid = cell.centroid
        point = Point(centroid.x, centroid.y)

        building_hits = len(building_tree.query(cell)) if building_tree else 0
        park_hits = len(park_tree.query(cell)) if park_tree else 0
        centroid_building_hits = len(building_tree.query(point)) if building_tree else 0
        centroid_park_hits = len(park_tree.query(point)) if park_tree else 0
        tree_count = _count_points(trees, point, radius=200.0)
        road_count = _count_points(road_points, point, radius=120.0)

        blocked = centroid_building_hits > 0 or centroid_park_hits > 0
        if blocked:
            feasibility = 0.0
        else:
            feasibility = max(0.0, 1.0 - min(road_count / 12.0, 1.0))

        raw_rows.append(
            {
                "cell_id": f"cell-{idx:03d}",
                "centroid_x": centroid.x,
                "centroid_y": centroid.y,
                "building_hits": float(building_hits),
                "tree_count": float(tree_count),
                "park_hits": float(park_hits),
                "road_count": float(road_count),
                "feasibility": feasibility,
            }
        )

    cooling_raw = [row["building_hits"] - 0.5 * row["tree_count"] for row in raw_rows]
    pollution_raw = [1.2 * row["road_count"] + 0.4 * row["building_hits"] - 0.35 * row["tree_count"] for row in raw_rows]
    equity_raw = [1.5 * (1.0 if row["park_hits"] == 0 else 0.0) + max(0.0, 6.0 - row["tree_count"]) for row in raw_rows]
    biodiversity_raw = [row["park_hits"] * 0.5 + row["tree_count"] * 0.2 for row in raw_rows]
    budget_raw = [sqrt(max(1.0, row["road_count"] + 1.0)) for row in raw_rows]

    cooling = _normalize(cooling_raw)
    pollution = _normalize(pollution_raw)
    equity = _normalize(equity_raw)
    biodiversity = _normalize(biodiversity_raw)
    budget_fit = _normalize(budget_raw, invert=True)

    results: list[CellScore] = []
    for row, cool, pol, eq, bio, budget in zip(raw_rows, cooling, pollution, equity, biodiversity, budget_fit):
        combined = (
            0.24 * cool
            + 0.22 * pol
            + 0.22 * eq
            + 0.15 * bio
            + 0.15 * row["feasibility"]
            + 0.02 * budget
        )
        results.append(
            CellScore(
                cell_id=row["cell_id"],
                centroid_x=row["centroid_x"],
                centroid_y=row["centroid_y"],
                cooling_need=round(cool, 4),
                pollution_need=round(pol, 4),
                equity_need=round(eq, 4),
                biodiversity_need=round(bio, 4),
                feasibility=round(row["feasibility"], 4),
                budget_fit=round(budget, 4),
                combined_score=round(combined, 4),
                evidence={
                    "nearby_buildings": int(row["building_hits"]),
                    "nearby_trees": int(row["tree_count"]),
                    "park_overlap": int(row["park_hits"]),
                    "centroid_blocked_by_building": int(centroid_building_hits),
                    "centroid_blocked_by_park": int(centroid_park_hits),
                    "nearby_road_nodes": int(row["road_count"]),
                    "pollution_pressure_proxy": round(pol, 4),
                },
            )
        )
    return sorted(results, key=lambda item: item.combined_score, reverse=True)
