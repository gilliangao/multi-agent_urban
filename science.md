# Urban Tree Canopy Cooling Model for Multi-Agent Optimisation

## Problem Statement

We model a simplified city as a **4 × 4 grid of urban blocks**.

The goal is to determine:

- how many trees should be planted
- which tree species should be used
- where trees should be planted

to minimise local urban temperatures while balancing:

- cooling performance
- biodiversity
- equity
- cost
- urban ventilation

This becomes a **multi-agent optimisation/debate problem**.

---

# Core Cooling Equation

A simplified empirically-based urban cooling equation:

\[
\boxed{
\Delta T_i
=
-0.04
\times
C_i
\times
S_i
\times
U_i
}
\]

Where:

| Variable | Meaning |
|---|---|
| \(\Delta T_i\) | Local temperature reduction for block \(i\) (°C) |
| \(C_i\) | Tree canopy cover percentage (%) |
| \(S_i\) | Tree species cooling factor |
| \(U_i\) | Urban morphology factor |

---

# Parameter Definitions

## 1. Canopy Cover (\(C\))

Tree canopy coverage percentage within a block.

Examples:

| Canopy Cover | Value |
|---|---:|
| 10% canopy | 10 |
| 20% canopy | 20 |
| 40% canopy | 40 |

Empirical studies suggest:

- every 10% increase in canopy cover may reduce local air temperature by:

\[
0.3^\circ C \text{ to } 0.7^\circ C
\]

depending on conditions.

---

## 2. Species Cooling Factor (\(S\))

Different tree species provide different cooling performance.

### Suggested values

| Tree Type | \(S\) |
|---|---:|
| Small ornamental tree | 0.7 |
| Medium deciduous tree | 1.0 |
| Large broadleaf shade tree | 1.4 |
| High-transpiration dense canopy tree | 1.6 |

Cooling performance depends on:

- canopy size
- leaf area index (LAI)
- evapotranspiration
- tree height
- foliage density

---

## 3. Urban Morphology Factor (\(U\))

Urban geometry affects cooling efficiency.

### Suggested values

| Urban Form | \(U\) |
|---|---:|
| Open park | 1.2 |
| Residential street | 1.0 |
| Dense downtown canyon | 0.7 |
| Poor ventilation zone | 0.5 |

Trees are generally less effective in highly dense urban canyons.

---

# Example Calculation

Suppose a city block has:

- 35% canopy cover
- large shade trees
- residential morphology

Then:

\[
\Delta T
=
-0.04
\times
35
\times
1.4
\times
1.0
\]

\[
\Delta T \approx -1.96^\circ C
\]

Therefore the block is predicted to be approximately:

\[
2^\circ C
\]

cooler during hot daytime conditions.

---

# Number of Trees Required

## Tree Number Equation

\[
\boxed{
N
=
\frac{
A_{city}
\times
C
}{
A_{canopy/tree}
}
}
\]

Where:

| Variable | Meaning |
|---|---|
| \(N\) | Number of trees required |
| \(A_{city}\) | Total city area |
| \(C\) | Target canopy fraction |
| \(A_{canopy/tree}\) | Average mature canopy area per tree |

---

# Example

Suppose:

- city area = 160,000 m²
- target canopy = 30%
- average mature canopy/tree = 50 m²

Then:

\[
N
=
\frac{
160000
\times
0.3
}{
50
}
=
960
\]

So approximately:

\[
960
\]

medium trees are required.

---

# Extended Urban Climate Equation

A more physically-based formulation:

\[
\boxed{
T_{local}
=
T_{base}
-
\alpha C
+
\beta I
-
\gamma ET
}
\]

Where:

| Variable | Meaning |
|---|---|
| \(T_{local}\) | Local urban temperature |
| \(T_{base}\) | Background city temperature |
| \(C\) | Canopy cover |
| \(I\) | Impervious surface fraction |
| \(ET\) | Evapotranspiration capacity |
| \(\alpha,\beta,\gamma\) | Empirical coefficients |

---

# Physical Cooling Mechanisms

Urban trees cool cities through:

## 1. Shade

Reduces incoming shortwave solar radiation.

## 2. Evapotranspiration

Trees evaporate water and consume latent heat:

\[
Q_{latent}
=
L_v E
\]

Where:

| Variable | Meaning |
|---|---|
| \(Q_{latent}\) | Cooling energy |
| \(L_v\) | Latent heat of vaporisation |
| \(E\) | Evapotranspiration rate |

Higher transpiration generally increases cooling.

---

# Multi-Agent Optimisation Framework

The city optimisation objective can be written as:

\[
\boxed{
\max
\left(
w_1 \Delta T
+
w_2 B
+
w_3 E
-
w_4 Cost
\right)
}
\]

Where:

| Variable | Meaning |
|---|---|
| \(\Delta T\) | Cooling benefit |
| \(B\) | Biodiversity score |
| \(E\) | Equity score |
| \(Cost\) | Planting + maintenance cost |
| \(w_i\) | Agent weighting factors |

---

# Example Agents

## Climate Agent

Goal:

\[
\max \Delta T
\]

Prefers:

- large-canopy trees
- high-transpiration species
- hottest urban blocks

---

## Budget Agent

Goal:

\[
\min \frac{Cost}{\Delta T}
\]

Prefers:

- efficient species
- fewer high-impact trees

---

## Biodiversity Agent

Goal:

\[
\max Diversity
\]

Prefers:

- mixed species planting
- ecological corridors

---

## Equity Agent

Goal:

\[
\max Fairness
\]

Prefers:

- equitable cooling distribution
- planting in underserved neighbourhoods

---

## Ventilation Agent

Goal:

Maintain airflow and reduce nighttime heat trapping.

Prefers:

- avoiding excessive canopy in dense street canyons

---

# Suggested Simulation Question

> “Given a 4 × 4 city grid and a target reduction in urban heat, where should different tree species be planted to maximise cooling while balancing biodiversity, equity, ventilation, and cost?”

---

# Literature-Based Empirical Benchmarks

## Reported cooling from urban tree canopy

| Study Result | Approximate Cooling |
|---|---:|
| 10% canopy increase | 0.3–0.7°C |
| 30% canopy increase | ~1.5°C |
| Strong daytime cooling cases | 1–5°C |

Cooling varies depending on:

- weather conditions
- urban density
- tree species
- canopy connectivity
- cloud cover
- time of day

---

# Key Insight

The most important finding from the literature:

> Urban cooling depends more on mature connected canopy structure than simply the number of trees.

Large interconnected canopies generally outperform isolated small trees.