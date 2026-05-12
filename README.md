# Emergent Canopy

Emergent Canopy is a research-oriented multi-agent urban climate planning project. It studies how competing specialist agents coordinate sequential tree-planting interventions over a city grid while preserving traceability, explainability, and evaluation rigor.

## Core Idea

The system treats urban greening as a scientific decision problem:

1. Load spatial evidence for a region.
2. Construct candidate planting cells.
3. Score each cell across multiple objectives.
4. Run a sequential coordination episode under a planting budget.
5. Run a shared-transcript multi-agent debate over the best interventions, including opening statements, rebuttals, and critique.
6. Save a traceable explanation bundle for every recommendation.

## Research Question

How can a multi-agent reinforcement learning style urban planning system coordinate competing stakeholder policies over sequential planting decisions to maximize shared urban climate benefit while preserving interpretability and traceability?

## Stack

- `LangGraph` for orchestration and stateful workflow control
- `Langfuse` hooks for observability when keys are configured
- `Denario` as a research-workflow inspiration and compatibility target
- `AG2` as a multi-agent dependency target, with runtime diagnostics
- `Streamlit` for the interactive demo layer

## Why This Is MARL-Style

The current system includes:

- multiple agents with distinct objective functions
- sequential allocation under a limited budget
- an environment state that changes after each selected cell
- shared reward and coordination metrics per round
- explicit conflict and coalition tracking

This is a lightweight MARL-style coordination scaffold, designed to support fuller policy learning in the next phase.

## Repository Structure

- `src/urban_forest_ai/`: application and research pipeline code
- `configs/`: experiment and runtime configuration
- `docs/`: architecture, roadmap, and research planning
- `prompts/`: prompt assets and agent instructions
- `scripts/`: utility scripts for setup and experiments
- `tests/`: test placeholders for future validation
- `data/`: local-only spatial parquet inputs
- `outputs/`: local-only traces, reports, and artifacts

## How the Prototype Works

This prototype has three user-facing entry points:

- `app.py`: the main Streamlit control room in the repo root
- `frontend/app.py`: a Streamlit frontend that uses the backend package in `backend/src/urban_forest_ai`
- `backend/src/urban_forest_ai/main.py`: a FastAPI app with `/health`, `/data/status`, `/areas`, and `/plan`

All of them eventually call the same planning pipeline:

1. `run_pipeline(...)` selects an area of interest from the boundary data.
2. The data loader reads local parquet inputs for buildings, parks, roads, trees, and boundaries.
3. The scoring layer builds a grid of candidate planting cells over the selected area.
4. Each cell receives numeric indicators for cooling, pollution, equity, biodiversity, feasibility, and budget fit.
5. A coordination episode runs under a fixed planting budget so the specialist agents can express conflicting preferences over multiple rounds.
6. The top cells then enter a shared-transcript debate where agents argue in natural language, revise or defend their positions, and produce final votes.
7. The explainability layer builds a reason chain, optionally asks an OpenAI model for a natural-language explanation, and saves a trace bundle to `outputs/`.

### Runtime Flow

The planning flow is split into two reasoning layers:

- **Coordination layer**
  - implemented in `src/urban_forest_ai/marl/`
  - simulates sequential planning under a limited budget
  - tracks coalition size, conflict rate, coordination score, and shared reward
  - answers: which cells do the agents align on over time?

- **Debate layer**
  - implemented in `src/urban_forest_ai/debate.py`
  - uses a LangGraph state graph
  - produces a shared transcript, agent votes, critique turns, and a judge summary
  - answers: why do the agents support, oppose, or hesitate on a given intervention?

### Debate Sequence

For each debated cell, the system currently runs:

1. `HeatAgent` opening
2. `PollutionAgent` opening
3. `EquityAgent` opening
4. `BiodiversityAgent` opening
5. `FeasibilityAgent` opening
6. `CritiqueAgent` opening
7. `HeatAgent` rebuttal
8. `PollutionAgent` rebuttal
9. `EquityAgent` rebuttal
10. `BiodiversityAgent` rebuttal
11. `FeasibilityAgent` rebuttal
12. `CritiqueAgent` rebuttal
13. `Judge` summary

Each specialist agent receives a narrow resource packet tied to its role rather than the full raw evidence blob. The `CritiqueAgent` is there specifically to make the debate more meaningful by challenging overconfidence, unresolved trade-offs, and weak assumptions before the judge aggregates the scores.

### OpenAI Usage and Fallback Mode

The debate agents and the explainer can call OpenAI models when `OPENAI_API_KEY` is available.

- Debate-stage agent turns use `OPENAI_DEBATE_MODEL`
- Explanation generation uses `OPENAI_EXPLAINER_MODEL`

If `OPENAI_API_KEY` is missing, the prototype still works by falling back to deterministic heuristics for the debate. Those fallback turns still populate the transcript so the control room remains usable. If you want strict failure when live model calls are unavailable, set:

```bash
export OPENAI_DEBATE_REQUIRED=1
```

### What Gets Produced

A successful planning run produces:

- a coordination episode summary
- ranked candidate cells
- per-agent final votes
- a shared debate transcript including critique turns
- a judge summary
- a reason chain
- an optional LLM explanation
- a JSON trace written to `outputs/<area>_trace.json`

## Data

The app expects these parquet files in `data/`:

- `E12000007_buildings.parquet`
- `E12000007_public_park_sites.parquet`
- `E12000007_road_nodes.parquet`
- `E12000007_road_edges.parquet`
- `E12000007_trees.parquet`
- `E12000007_boundaries.parquet`

These files are intentionally not tracked in git because they are too large for GitHub. Place them in `data/` locally before running the app.

## Run the CLI

```bash
.venv312/bin/python run_demo.py --area Westminster --cell-size 150 --top-k 5
```

## Run the UI

```bash
.venv312/bin/streamlit run app.py
```

## GitHub Secrets

Add only this secret in your GitHub repository settings:

- `OPENAI_API_KEY`

Path in GitHub:

- `Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

Then create:

- `Name`: `OPENAI_API_KEY`
- `Secret`: your OpenAI API key

The app reads `OPENAI_API_KEY` from the environment, so in GitHub Actions it should be exposed as:

```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Model Configuration

Model names are not secrets. Keep them configurable through normal environment variables.

Current model configuration:

- `OPENAI_DEBATE_MODEL`
- `OPENAI_EXPLAINER_MODEL`

Example:

```bash
export OPENAI_DEBATE_MODEL=gpt-5-mini
export OPENAI_EXPLAINER_MODEL=gpt-5-mini
```

If unset, the code defaults to:

- `gpt-5-mini` for debate-stage agent turns
- `gpt-5-mini` for the explanation layer

## Key Docs

- [Architecture Plan](./docs/ARCHITECTURE.md)
- [Research Roadmap](./docs/RESEARCH_PLAN.md)
- [System Diagram](./docs/SYSTEM_DIAGRAM.md)
- [Experiment Config](./configs/base_experiment.yaml)

## Implemented Agents

The current branch implements these agents:

- `HeatAgent`: prioritizes cooling impact in dense built areas
- `PollutionAgent`: prioritizes pollution reduction in road-heavy and dense urban cells
- `EquityAgent`: prioritizes underserved areas with low green access
- `BiodiversityAgent`: prioritizes ecological connectivity and green structure
- `FeasibilityAgent`: prioritizes deployability under infrastructure and budget constraints
- `CritiqueAgent`: challenges weak assumptions and unresolved trade-offs in the recommendation
- `Judge`: aggregates debate signals into a final recommendation

The coordination episode also exposes how these agents align or conflict across sequential planting rounds under a limited budget.

### Agent Table

| Agent | Objective | Inputs | Outputs |
| --- | --- | --- | --- |
| `HeatAgent` | Maximize cooling benefit | `cooling_need`, nearby buildings, nearby trees | preferred cell in coordination, support/oppose vote in debate |
| `PollutionAgent` | Reduce local pollution exposure | `pollution_need`, nearby road nodes, pollution proxy | preferred cell in coordination, support/oppose vote in debate |
| `EquityAgent` | Improve environmental fairness | `equity_need`, park overlap, tree scarcity proxy | preferred cell in coordination, support/oppose vote in debate |
| `BiodiversityAgent` | Improve ecological connectivity | `biodiversity_need`, existing green structure | preferred cell in coordination, support/oppose vote in debate |
| `FeasibilityAgent` | Keep the plan realistic and deployable | `feasibility`, `budget_fit`, road/building constraints | preferred cell in coordination, support/oppose vote in debate |
| `CritiqueAgent` | Stress-test weak assumptions and unresolved risk | cross-agent scores plus practical/equity risk indicators | critique-stage support/oppose vote in debate |
| `Judge` | Aggregate debate-stage signals | all agent votes plus cell indicator scores | final debate summary and final score |

### Agent Roles and Implementation

- `HeatAgent`
  - Goal: improve urban cooling.
  - It prefers cells with high `cooling_need`.
  - It is implemented in the debate layer and in the coordination episode preference function.

- `PollutionAgent`
  - Goal: reduce pollution exposure.
  - It prefers cells with high `pollution_need`.
  - This indicator is driven by road exposure, dense urban pressure, and lack of trees.

- `EquityAgent`
  - Goal: improve environmental access in underserved areas.
  - It prefers cells with high `equity_need`.
  - This captures where green benefit is most lacking.

- `BiodiversityAgent`
  - Goal: improve ecological connectivity and green-network quality.
  - It prefers cells with high `biodiversity_need`.

- `FeasibilityAgent`
  - Goal: keep the plan implementable.
  - It prefers cells with high combined feasibility and budget fit.

- `CritiqueAgent`
  - Goal: challenge overconfidence and unresolved contradictions.
  - It reads the shared debate transcript and pressures the rest of the panel to justify their trade-offs.

- `Judge`
  - Goal: summarize the debate-stage decision.
  - It aggregates the agent votes and reports the final recommendation in a human-readable way.

## How Coordination Works

Coordination is implemented as a lightweight MARL-style episode over a limited planting budget.

1. The system starts with a set of candidate cells and a fixed planting budget.
2. In each round, every agent selects its preferred remaining cell using its own objective.
3. These preferences are compared to form an implicit coalition.
4. The winning cell is chosen by strongest support, with ties resolved using cell quality and feasibility.
5. The environment updates by removing that cell from the remaining candidate set.
6. A shared reward is computed from cooling, pollution, equity, biodiversity, and feasibility.
7. The next round begins with the updated state.

### Coordination Metrics

- `coalition_size`: how many agents aligned on the chosen cell in that round
- `coordination_score`: coalition size divided by the number of agents
- `conflict_rate`: how much disagreement occurred across the episode
- `shared_reward`: how strong the selected intervention was from the system-wide perspective

### Coordination vs Debate

The system has two distinct layers:

- **Coordination layer**
  - sequential planning under a limited planting budget
  - implemented in `marl.py`
  - answers: which actions do the agents coordinate on over time?

- **Debate layer**
  - richer per-cell votes, shared transcript turns, and explanations
  - implemented in `debate.py` using LangGraph
  - answers: why did the agents support or oppose this intervention?

### Debate Mechanics

- Each specialist agent receives a narrow resource packet tied to its objective rather than the full raw evidence blob.
- The debate now runs as a shared transcript with opening statements, rebuttals, and a dedicated critique pass.
- Agents try OpenAI-backed turns first and fall back to deterministic heuristics when `OPENAI_API_KEY` is unavailable, unless `OPENAI_DEBATE_REQUIRED=1`.

## Launch from GitHub Actions

There is a manual workflow at [`.github/workflows/launch-control-room.yml`](/Users/carlamuntean/climate-council/.github/workflows/launch-control-room.yml) that starts the Streamlit control room on a GitHub runner and exposes a temporary public URL through Cloudflare Tunnel.

After triggering the workflow from the Actions tab, open the job summary to get the temporary frontend URL. The run keeps the tunnel alive for about 55 minutes.

## Notes on Denario and AG2

`denario` and `ag2` are part of the intended stack, but their current published combination hits an upstream `autogen.llm_config` circular-import failure in this environment.

Because of that, the branch currently runs through the compatible internal coordination/debate implementation rather than executing Denario or AG2 directly.

What is still preserved from that direction:

- runtime diagnostics
- Denario prompt lineage
- a Denario-inspired scientific workflow framing
- a compatible fallback path for the coordination and debate engine

## Demo Framing

Use this framing in a presentation:

> Emergent Canopy is a Denario-inspired multi-agent urban climate research system that learns and debates intervention priorities over a city grid, with explicit evidence, per-agent reasoning, and auditable output traces.
