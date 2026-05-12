from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely.geometry import box


def _default_data_dir() -> Path:
    env_value = os.getenv("EMERGENT_CANOPY_DATA_DIR")
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parents[3] / "data"


DATA_DIR = _default_data_dir()
REQUIRED_DATA_FILES = [
    "E12000007_buildings.parquet",
    "E12000007_public_park_sites.parquet",
    "E12000007_road_nodes.parquet",
    "E12000007_road_edges.parquet",
    "E12000007_trees.parquet",
    "E12000007_boundaries.parquet",
]


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def missing_data_files() -> list[str]:
    ensure_data_dir()
    return [name for name in REQUIRED_DATA_FILES if not (DATA_DIR / name).exists()]


@lru_cache(maxsize=1)
def available_area_names() -> list[str]:
    if (DATA_DIR / "E12000007_boundaries.parquet").exists():
        boundaries = _read_boundary_frame()
        names = sorted({str(name) for name in boundaries["LSOA21NM"].dropna().tolist()})
        return names
    return []


def _read_boundary_frame() -> pd.DataFrame:
    path = DATA_DIR / "E12000007_boundaries.parquet"
    return pd.read_parquet(path, columns=["MSOA21NM", "LAD22NM", "geometry"])


def list_msoas() -> list[str]:
    df = pd.read_parquet(DATA_DIR / "E12000007_boundaries.parquet", columns=["MSOA21NM"])
    return sorted(df["MSOA21NM"].dropna().unique().tolist())


def list_lads() -> list[str]:
    df = pd.read_parquet(DATA_DIR / "E12000007_boundaries.parquet", columns=["LAD22NM"])
    return sorted(df["LAD22NM"].dropna().unique().tolist())


def select_area(area_query: str | None = None, level: str = "msoa") -> tuple[str, object]:
    col = "MSOA21NM" if level == "msoa" else "LAD22NM"
    boundaries = _read_boundary_frame()
    if area_query:
        mask = boundaries[col].str.contains(area_query, case=False, na=False)
        subset = boundaries[mask]
        if subset.empty:
            raise ValueError(f"No {level.upper()} matched query: {area_query!r}")
        area_name = str(subset.iloc[0][col])
    else:
        area_name = str(boundaries.iloc[0][col])

    subset = boundaries[boundaries[col] == area_name]
    geoms = shapely.from_wkb(subset["geometry"].dropna().tolist())
    polygon = shapely.unary_union(geoms)
    return area_name, polygon


def load_parks(aoi) -> list[object]:
    frame = pd.read_parquet(DATA_DIR / "E12000007_public_park_sites.parquet", columns=["geometry"])
    geoms = shapely.from_wkb(frame["geometry"].dropna().tolist())
    clipped = [geom for geom in geoms if geom.intersects(aoi)]
    return clipped


def load_road_nodes(aoi) -> np.ndarray:
    frame = pd.read_parquet(DATA_DIR / "E12000007_road_nodes.parquet", columns=["x", "y"])
    minx, miny, maxx, maxy = aoi.bounds
    subset = frame[
        frame["x"].between(minx, maxx)
        & frame["y"].between(miny, maxy)
    ]
    return subset[["x", "y"]].to_numpy(dtype=float)


def _chunked_wkb_filter(path: Path, aoi, geometry_column: str = "geometry") -> list[object]:
    bbox = box(*aoi.bounds)
    parquet_file = pq.ParquetFile(path)
    kept: list[object] = []
    for idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(idx, columns=[geometry_column])
        values = table.column(0).to_pylist()
        geoms = shapely.from_wkb([value for value in values if value is not None])
        kept.extend([geom for geom in geoms if geom.intersects(bbox)])
    return kept


def load_buildings(aoi) -> list[object]:
    return _chunked_wkb_filter(DATA_DIR / "E12000007_buildings.parquet", aoi)


def load_trees(aoi) -> np.ndarray:
    bbox = box(*aoi.bounds)
    parquet_file = pq.ParquetFile(DATA_DIR / "E12000007_trees.parquet")
    points: list[tuple[float, float]] = []
    for idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(idx, columns=["geometry"])
        values = table.column(0).to_pylist()
        geoms = shapely.from_wkb([value for value in values if value is not None])
        for geom in geoms:
            if geom is not None and geom.intersects(bbox):
                points.append((geom.x, geom.y))
    if not points:
        return np.empty((0, 2))
    return np.asarray(points, dtype=float)
