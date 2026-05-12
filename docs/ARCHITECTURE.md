# Architecture Plan

## Goal

Build a frontier-feeling research system for urban climate intervention planning that is:

- multi-agent
- explainable
- traceable
- experimentally extensible
- compatible with future MARL-style extensions

## System Layers

### 1. Data Layer

Responsibilities:

- load local parquet datasets
- define the area of interest
- expose geometry and feature primitives to the rest of the system

Current implementation:

- `src/urban_forest_ai/data_loader.py`

Future extensions:

- cached feature extraction
- precomputed spatial indexes
- support for additional city regions

### 2. Environment Layer

Responsibilities:

- define candidate planting cells
- define state variables for each cell
- represent intervention constraints and resource budgets
- evolve the environment across sequential planting rounds

Current implementation:

- `src/urban_forest_ai/scoring.py`

Current MARL-style behavior:

- multi-step coordination episode
- limited planting budget
- state changes as cells are allocated
- shared reward and coordination metrics per round

Future MARL direction:

- formalized action space beyond cell selection
- repeated episodes with policy updates
- budget-aware sequential planting campaigns

### 3. Agent Layer

Responsibilities:

- represent competing planning priorities
- produce local judgments over candidate cells
- expose human-readable rationales for every decision

Current agents:

- `HeatAgent`
- `EquityAgent`
- `BiodiversityAgent`
- `FeasibilityAgent`
- `Judge`

Current implementation:

- `src/urban_forest_ai/debate.py`

Future extensions:

- explicit policy memory
- iterative negotiation rounds
- lightweight learning updates per episode

### 4. Orchestration Layer

Responsibilities:

- manage workflow order
- preserve state between stages
- make the system easy to test and extend

Current implementation:

- `src/urban_forest_ai/pipeline.py`
- `LangGraph` state graph in `src/urban_forest_ai/debate.py`

Future extensions:

- experiment runners
- branching flows for ablations
- human-in-the-loop overrides

### 5. Explainability Layer

Responsibilities:

- store evidence per recommendation
- preserve full reason chains
- provide optional natural-language summaries
- serialize traces to disk

Current implementation:

- `src/urban_forest_ai/explainability.py`

Outputs:

- score evidence
- agent votes
- judge summary
- reason chain
- optional LLM explanation
- trace JSON bundle

### 6. Interface Layer

Responsibilities:

- provide research CLI
- provide demo UI
- make outputs visible to judges and collaborators

Current implementation:

- `run_demo.py`
- `app.py`

Future extensions:

- scenario explorer
- policy comparison dashboard
- episode replay visualizer

## Research Workflow

1. Prepare or copy local datasets into `data/`
2. Select an LSOA or area query
3. Generate candidate cells
4. Score cells across planning dimensions
5. Run a sequential coordination episode under a planting budget
6. Run agent debate over top candidates
7. Save explanations and trace outputs
8. Compare against baselines

## Evaluation Framework

Primary metrics:

- cooling proxy improvement
- equity proxy improvement
- biodiversity/connectivity proxy
- feasibility/cost proxy
- debate score
- coordination score
- conflict rate
- shared reward

Baseline comparisons:

- greedy score ranking
- low-score reference
- future random baseline
- future heuristic budgeted baseline

## MARL Upgrade Path

The current system is now coordination-first with a lightweight MARL-style episode layer, but it is not yet full learned MARL. To evolve it into a more authentic MARL research system:

1. Introduce explicit environment state transitions
2. Move from single-pass ranking to sequential episodes
3. Give each agent action choices instead of only votes
4. Add individual and shared reward signals
5. Measure convergence and coordination dynamics over multiple episodes

This repo structure is designed so those changes fit naturally rather than requiring a rewrite.
