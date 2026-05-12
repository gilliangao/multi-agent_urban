# Research Plan

## Research Question

How can a multi-agent reinforcement learning style urban planning system coordinate competing stakeholder policies over sequential planting decisions to maximize shared urban climate benefit while preserving interpretability and traceability?

## Hypothesis

A sequential coordination system with agent-specific objectives and a shared reward signal will produce more balanced planting policies, higher coordination quality, and better-explained interventions than a single-score greedy baseline.

## Current MVP

- candidate generation over a real spatial grid
- deterministic feature scoring
- sequential coordination episode under a planting budget
- multi-agent debate over top cells
- evidence and trace export

## Next Milestones

### Milestone 1: Robust Research Demo

- stabilize data ingestion
- harden CLI and UI flows
- add experiment configs
- improve trace exports

### Milestone 2: Stronger Evaluation

- add random and heuristic baselines
- produce area-level aggregate metrics
- add reproducible experiment outputs

### Milestone 3: Stronger MARL Core

- sequential planting episodes
- explicit budget constraints per episode
- agent memory across rounds
- policy updates from episode outcomes
- convergence plots and coordination analysis

### Milestone 4: Stronger Scientific Positioning

- scenario sweeps
- sensitivity analysis
- policy-weight counterfactuals
- agent disagreement analysis

## Research Outputs

- ranked intervention recommendations
- evidence-backed justifications
- trace bundles
- baseline comparison tables
- coordination episode summaries
- future episode-level reward curves

## Current MARL Interpretation

The current system should be described as a lightweight MARL-style coordination environment rather than a fully trained deep MARL system.

It already includes:

- multiple agents with distinct objective functions
- sequential selection under a limited budget
- an environment state that changes as cells are allocated
- a shared reward per coordination step
- explicit coordination and conflict metrics

The next stage is to add learning updates across repeated episodes so the project moves from MARL-style coordination to clearer policy learning.

## Risks

- large local datasets make reproducibility harder
- Denario/AG2 runtime instability limits direct integration today
- current cell scoring is proxy-based rather than physically simulated

## Mitigations

- keep all local-data assumptions documented
- preserve fallback paths and runtime diagnostics
- separate research framing from hard implementation claims
