# Fire Risk & Simulation System

A two-module system for **optimally placing objects in a grid** to minimize fire damage, and **simulating fire spread** to analyze risk.

---

## Table of Contents

- [Overview](#overview)
- [Module 1 — Branch-and-Bound Placement Optimizer](#module-1--branch-and-bound-placement-optimizer)
  - [Goal](#goal)
  - [Precompute](#precompute)
  - [DFS with Branch-and-Bound](#dfs-with-branch-and-bound)
  - [Pruning Strategies](#pruning-strategies)
- [Module 2 — Fire Simulator](#module-2--fire-simulator)
  - [simulate_from_source — Deterministic Fire Spread](#simulate_from_source--deterministic-fire-spread)
  - [simulate_uniform — Expected-Value Fire Spread](#simulate_uniform--expected-value-fire-spread)
  - [How They Relate](#how-they-relate)

---

## Overview

The system answers two questions:

1. **Where should I place objects** in a grid to minimize expected loss if a fire breaks out?
2. **If a fire starts, what happens?** How does it spread, and what gets destroyed?

Expected loss for any object placed in a cell is defined as:

```
loss = (value × flammability / burn_resistance) × cell_risk
```

---

## Module 1 — Branch-and-Bound Placement Optimizer

**File:** `branch_and_bound.py`

### Goal

Assign objects to grid cells to **minimize total expected loss** across all objects.

### Precompute

```
for each cell in the n×m grid:
    centrality = how close it is (on average) to all other cells
    risk = 0.2 + 0.8 × centrality        // central cells are riskier

for each object:
    coeff = value × flammability / (1 + burn_time)   // loss-per-unit-risk

sort objects by coeff descending            // high-impact objects first
```

> **Why sort descending?** Placing high-impact objects first means the pruning bound bites earlier, cutting off more of the search tree.

### DFS with Branch-and-Bound

```
dfs(obj_index, remaining_cells, current_loss, assignment):

    if all objects placed:
        update best_loss and best_assignment if current_loss is better
        return

    // Pruning: compute an optimistic (lowest possible) future loss
    //   pair highest remaining coeffs with lowest remaining risks
    lower_bound = current_loss + sum(coeff[i] × risk[i])
                                  for i in min(remaining_objs, remaining_cells)
                                  sorted optimistically

    if lower_bound >= best_loss:
        return                              // prune this branch

    // Try placing current object into each remaining cell
    // sorted safest-first (lowest risk) to find good solutions early
    for cell in remaining_cells sorted by risk ascending:
        assign object → cell
        dfs(obj_index + 1,
            remaining_cells − {cell},
            current_loss + coeff × cell.risk,
            assignment)
        unassign object
```

### Pruning Strategies

Two ideas work together to avoid exploring large parts of the search space:

| Technique | How it works |
|---|---|
| **Optimistic bound** | Pairs the highest remaining `coeff` values with the lowest remaining `risk` values — the best-case future. If even this ideal pairing can't beat the current best solution, the branch is pruned. |
| **Safest-cell-first ordering** | Tries low-risk cells first, so strong (low-loss) incumbents are found early. This tightens the bound quickly, making subsequent pruning more aggressive. |

---

## Module 2 — Fire Simulator

**File:** `fire_simulator.py`

Two functions with different assumptions about where the fire starts.

---

### `simulate_from_source` — Deterministic Fire Spread

A single fire starts at one known source cell and spreads outward tick by tick.

```
setup:
    ignite source cell at t = 0
    object_exposure[obj] = 0 for all objects

for each timestep t:

    // Spread fire
    for each currently burning cell:
        for each neighbor cell (up/down/left/right):
            if not burning and not burnt out:
                ignite it, record ignite_time

    // Burn out cells that have been burning long enough
    for each burning cell:
        if (t - ignite_time) >= cell_burn_duration:
            mark as burnt_out (no longer burning)

    // Update object destruction
    for each object not yet destroyed:
        if object's cell is burning:
            exposure += dt
        if exposure >= burn_time:
            destroy object, add value to cumulative_loss

    record SimulationStep snapshot
```

**Key behavior:** fire spreads **one cell per tick** in 4 directions (up/down/left/right). Cells burn for `cell_burn_duration` then go cold and can no longer spread fire.

Each `SimulationStep` snapshot contains:

| Field | Description |
|---|---|
| `t` | Current time |
| `fire_front` | Cells currently burning |
| `newly_ignited` | Cells that caught fire this tick |
| `newly_destroyed` | Objects destroyed this tick |
| `cumulative_loss` | Total value lost so far |
| `fire_grid` | Full cell-state snapshot |

---

### `simulate_uniform` — Expected-Value Fire Spread

Assumes the fire source is **equally likely to start anywhere**. Runs `simulate_from_source` once per possible source cell (all `n×m` of them), then aggregates the results into probability estimates.

```
// Phase 1 — run all simulations
for each cell (r, c) as source:
    run simulate_from_source → one full run of steps

// Phase 2 — aggregate across all n×m runs
for each timestep i:
    for each cell (r, c):
        P(burning)   = count of runs where cell is burning   / total runs
        P(burnt_out) = count of runs where cell is burnt_out / total runs

    for each object:
        P(destroyed by t) = count of runs where obj destroyed by t / total runs

// Phase 3 — build consensus snapshot using 0.5 threshold
    fire_front     = cells where P(burning) >= 0.5
    newly_ignited  = cells that crossed the 0.5 threshold this tick
    expected_loss  = sum over objects of (value × P(destroyed))
```

In addition to the standard `SimulationStep` fields, each step also carries:

| Field | Description |
|---|---|
| `expected_fire_grid[r][c]` | Probability that cell `(r, c)` is burning at this timestep |
| `expected_loss_by_obj[id]` | Probability that object `id` has been destroyed by this timestep |

---

### How They Relate

```
simulate_uniform
    └─ calls simulate_from_source  n×m times
         (once per possible fire origin)
    └─ averages the results into probability fields
         → expected_fire_grid[r][c]   =  P(cell is burning at time t)
         → expected_loss_by_obj[id]   =  P(object is destroyed by time t)
```

| Function | Fire origin | Output type |
|---|---|---|
| `simulate_from_source` | One specific cell | Single deterministic scenario |
| `simulate_uniform` | All cells equally likely | Risk-weighted expected outcomes |