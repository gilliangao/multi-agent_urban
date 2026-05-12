# System Diagram

```mermaid
flowchart TD
    A["Local Spatial Data<br/>buildings, roads, parks, trees, boundaries"] --> B["Data Loader<br/>data_loader.py"]
    B --> C["Grid + Feature Scoring<br/>scoring.py"]
    C --> D["Candidate Cells"]

    D --> E["Coordination Episode<br/>marl.py"]
    E --> E1["HeatAgent preference"]
    E --> E2["EquityAgent preference"]
    E --> E3["BiodiversityAgent preference"]
    E --> E4["FeasibilityAgent preference"]
    E1 --> E5["Coalition / Conflict Resolution"]
    E2 --> E5
    E3 --> E5
    E4 --> E5
    E5 --> E6["Sequential cell allocation<br/>shared reward + coordination metrics"]

    D --> F["Debate Graph<br/>debate.py via LangGraph"]
    F --> F1["HeatAgent vote"]
    F --> F2["EquityAgent vote"]
    F --> F3["BiodiversityAgent vote"]
    F --> F4["FeasibilityAgent vote"]
    F1 --> G["Judge"]
    F2 --> G
    F3 --> G
    F4 --> G

    E6 --> H["Pipeline Orchestrator<br/>pipeline.py"]
    G --> H
    H --> I["Explainability Layer<br/>explainability.py"]
    I --> J["Reason chain"]
    I --> K["Optional OpenAI explanation"]
    I --> L["Trace JSON"]

    H --> M["CLI<br/>run_demo.py"]
    H --> N["Streamlit UI<br/>app.py"]
```

## Execution Stages

1. `data_loader.py` loads and filters spatial evidence for the chosen area.
2. `scoring.py` converts geometry into candidate grid cells and multi-objective feature scores.
3. `marl.py` runs a sequential coordination episode under a planting budget.
4. `debate.py` runs a LangGraph-based vote and judge pass over top candidates.
5. `pipeline.py` merges coordination and debate outputs.
6. `explainability.py` produces reason chains, optional LLM explanations, and trace outputs.
7. `run_demo.py` and `app.py` expose the system through CLI and UI.
