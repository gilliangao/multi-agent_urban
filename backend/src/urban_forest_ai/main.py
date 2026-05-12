from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data_loader import available_area_names, ensure_data_dir, missing_data_files
from .pipeline import run_pipeline
from .settings import get_model, get_secret


app = FastAPI(
    title="Emergent Canopy Backend",
    description="Backend API for live multi-agent urban climate planning sessions.",
    version="0.1.0",
)


class PlanRequest(BaseModel):
    area_query: str | None = Field(default="Westminster")
    cell_size: float = Field(default=150.0, ge=50.0, le=1000.0)
    top_k: int = Field(default=5, ge=1, le=20)
    budget: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debate/status")
def debate_status() -> dict[str, Any]:
    api_key = get_secret("OPENAI_API_KEY")
    return {
        "llm_mode": bool(api_key),
        "debate_model": get_model("OPENAI_DEBATE_MODEL", "gpt-4o-mini"),
        "explainer_model": get_model("OPENAI_EXPLAINER_MODEL", "gpt-4o-mini"),
    }


@app.get("/data/status")
def data_status() -> dict[str, Any]:
    data_dir = ensure_data_dir()
    missing = missing_data_files()
    return {
        "data_dir": str(data_dir.resolve()),
        "ready": len(missing) == 0,
        "missing_files": missing,
    }


@app.get("/areas")
def list_areas(query: str = "", limit: int = 20) -> dict[str, Any]:
    names = available_area_names()
    query_lower = query.lower()
    if query_lower:
        names = [name for name in names if query_lower in name.lower()]
    return {"areas": names[:limit]}


@app.post("/plan")
def plan(request: PlanRequest) -> dict[str, Any]:
    result = run_pipeline(
        area_query=request.area_query,
        cell_size=request.cell_size,
        top_k=request.top_k,
        budget=request.budget,
    )
    return {
        "area_name": result.area_name,
        "denario_status": result.denario_status,
        "candidates_considered": result.candidates_considered,
        "baseline_summary": result.baseline_summary,
        "coordination_episode": None if result.coordination_episode is None else asdict(result.coordination_episode),
        "trace_path": result.trace_path,
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
            for item in result.ranked_cells
        ],
    }
