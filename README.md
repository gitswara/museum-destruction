Core Algorithm: Branch-and-Bound Assignment (what branch_and_bound.py does)
The goal is to assign objects to grid cells to minimize total expected loss, where loss = (value × flammability / burn_resistance) × cell_risk.

Precompute
for each cell in the n×m grid:
    centrality = how close it is (on average) to all other cells
    risk = 0.2 + 0.8 × centrality        // central cells are riskier

for each object:
    coeff = value × flammability / (1 + burn_time)   // loss-per-unit-risk

sort objects by coeff descending            // high-impact objects first

DFS with Branch-and-Bound
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

Two key pruning ideas working together
TechniqueHowOptimistic boundPairs best coeffs with cheapest risks — if even this ideal future can't beat the current best, pruneSafest-cell-first ordering




Two Simulation Algorithms (what fire_simulator.py does)

simulate_from_source — Deterministic Fire Spread
A single fire starts at one source cell and spreads outward tick by tick.
setup:
    ignite source cell at t=0
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
Key behavior: fire spreads one cell per tick in 4 directions, cells burn for cell_burn_duration then go cold.

simulate_uniform — Expected-Value Fire Spread
Assumes the fire source is equally likely to start anywhere. Runs simulate_from_source once per possible source cell (all n×m of them), then aggregates.
// Monte Carlo over all possible ignition points
for each cell (r, c) as source:
    run simulate_from_source → one full run of steps

// Aggregate across all n×m runs
for each timestep i:
    for each cell (r, c):
        P(burning) = count of runs where cell is burning / total runs
        P(burnt_out) = count of runs where cell is burnt_out / total runs

    for each object:
        P(destroyed by t) = count of runs where obj destroyed by t / total runs

// Build consensus snapshot
    fire_front     = cells where P(burning) >= 0.5
    newly_ignited  = cells that crossed the 0.5 threshold this tick
    expected_loss  = sum over objects of (value × P(destroyed))

How they relate
simulate_uniform
    └─ calls simulate_from_source n×m times
         (once per possible fire origin)
    └─ averages the results into probability fields
         → expected_fire_grid[r][c]  = P(cell is burning at time t)
         → expected_loss_by_obj[id]  = P(object is destroyed by time t)
The first gives you a single deterministic scenario; the second gives you risk-weighted expected outcomes across all scenarios.