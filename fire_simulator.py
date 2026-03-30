from __future__ import annotations

"""Fire spread simulation primitives used by the UI and Flask endpoints.

This module exposes two public simulation entry points:

1) `simulate_from_source`: deterministic simulation from a single ignition cell.
2) `simulate_uniform`: expected simulation where every cell is treated as an
   equally likely ignition source and runs are aggregated per time step.

The code intentionally keeps the model simple:
- Fire spreads to 4-neighbors (up/down/left/right) each tick.
- A burning cell burns for `cell_burn_duration`, then becomes burnt out.
- Objects accumulate burn exposure while their cell is burning.
- An object is destroyed once exposure >= its `burn_time`, contributing `value`
  to cumulative loss.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class CellState:
    """State of a single grid cell at a specific time."""

    is_burning: bool = False
    is_burnt_out: bool = False


@dataclass
class PlacedObject:
    """Object anchored to a grid location with destruction attributes."""

    obj_id: str
    row: int
    col: int
    value: float
    burn_time: float


@dataclass
class SimulationStep:
    """Snapshot of the simulation at one discrete time value.

    Fields:
    - `t`: simulation time.
    - `fire_front`: all currently burning cells.
    - `newly_ignited`: cells that started burning at this step.
    - `newly_destroyed`: object IDs destroyed at this step.
    - `cumulative_loss`: total value lost up to and including this step.
    - `fire_grid`: full per-cell state.
    - `expected_fire_grid`: only for uniform mode (burning probabilities).
    - `expected_loss_by_obj`: only for uniform mode (destruction probability).
    """

    t: float
    fire_front: List[Tuple[int, int]]
    newly_ignited: List[Tuple[int, int]]
    newly_destroyed: List[str]
    cumulative_loss: float
    fire_grid: List[List[CellState]]
    expected_fire_grid: Optional[List[List[float]]] = None
    expected_loss_by_obj: Optional[Dict[str, float]] = None


def _neighbor_cells(r: int, c: int, n: int, m: int) -> Iterable[Tuple[int, int]]:
    """Yield valid 4-neighbors of `(r, c)` within `n x m` bounds."""

    if r > 0:
        yield (r - 1, c)
    if r < n - 1:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c < m - 1:
        yield (r, c + 1)


def _clone_grid(grid: List[List[CellState]]) -> List[List[CellState]]:
    """Return a deep-ish copy of cell flags for snapshotting step history."""

    return [[CellState(cell.is_burning, cell.is_burnt_out) for cell in row] for row in grid]


def _parse_objects(objects_at_cells: List[Dict[str, Any]], n: int, m: int) -> List[PlacedObject]:
    """Validate and normalize object dictionaries into `PlacedObject` records."""

    parsed: List[PlacedObject] = []
    for idx, item in enumerate(objects_at_cells):
        obj_id = str(item.get("obj_id", f"obj_{idx + 1}"))
        row = int(item["row"])
        col = int(item["col"])
        if not (0 <= row < n and 0 <= col < m):
            raise ValueError(f"Object {obj_id} is out of bounds: ({row}, {col})")

        parsed.append(
            PlacedObject(
                obj_id=obj_id,
                row=row,
                col=col,
                value=float(item["value"]),
                burn_time=max(0.0, float(item["burn_time"])),
            )
        )

    return parsed


def simulate_from_source(
    n: int,
    m: int,
    cell_burn_duration: float,
    max_time: float,
    dt: float,
    source_r: int,
    source_c: int,
    objects_at_cells: List[Dict[str, Any]],
) -> List[SimulationStep]:
    """Run a deterministic fire simulation from a fixed source cell.

    The update order at each time step is:
    1) Spread from currently burning cells to neighbors.
    2) Burn out cells whose burn duration elapsed.
    3) Update object exposure/destruction and cumulative loss.
    4) Emit a full snapshot (`SimulationStep`).
    """

    # Defensive input checks keep API errors deterministic and readable.
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if cell_burn_duration <= 0:
        raise ValueError("cell_burn_duration must be > 0")
    if max_time < 0:
        raise ValueError("max_time must be >= 0")
    if not (0 <= source_r < n and 0 <= source_c < m):
        raise ValueError("source cell is out of bounds")

    objects = _parse_objects(objects_at_cells, n, m)

    # `fire_grid` is the mutable in-place state for the active simulation run.
    fire_grid = [[CellState() for _ in range(m)] for _ in range(n)]
    # Maps each ignited cell to the time it started burning.
    ignite_time: Dict[Tuple[int, int], float] = {}
    # Accumulated burning time per object (seconds/time-units of exposure).
    object_exposure: Dict[str, float] = {obj.obj_id: 0.0 for obj in objects}
    # Object IDs already destroyed; prevents duplicate loss accounting.
    destroyed: set[str] = set()
    cumulative_loss = 0.0

    def ignite_cell(r: int, c: int, t: float) -> bool:
        """Ignite a cell once; returns True only if ignition actually happened."""

        cell = fire_grid[r][c]
        if cell.is_burning or cell.is_burnt_out:
            return False
        cell.is_burning = True
        ignite_time[(r, c)] = t
        return True

    initial_new_ignitions: List[Tuple[int, int]] = []
    if ignite_cell(source_r, source_c, 0.0):
        initial_new_ignitions.append((source_r, source_c))

    # Build explicit time points to avoid floating-point drift in `range` logic.
    times: List[float] = []
    t_cursor = 0.0
    while t_cursor <= max_time + 1e-12:
        times.append(round(t_cursor, 10))
        t_cursor += dt

    # Ordered simulation snapshots that API/UI consume for playback.
    steps: List[SimulationStep] = []

    for step_idx, t in enumerate(times):

        if step_idx == 0:
            # Step zero only contains the initial source ignition.
            newly_ignited = initial_new_ignitions
        else:
            # Capture current burning cells before we mutate state via spread.
            burning_before_tick = {
                (r, c)
                for r in range(n)
                for c in range(m)
                if fire_grid[r][c].is_burning
            }

            newly_ignited = []
            for r, c in burning_before_tick:
                for nr, nc in _neighbor_cells(r, c, n, m):
                    if ignite_cell(nr, nc, t):
                        newly_ignited.append((nr, nc))

        # Burnout update: cells stop burning and switch to burnt-out once their
        # individual burn duration has elapsed.
        for (r, c), start_t in list(ignite_time.items()):
            if fire_grid[r][c].is_burning and (t - start_t) >= cell_burn_duration:
                fire_grid[r][c].is_burning = False
                fire_grid[r][c].is_burnt_out = True

        newly_destroyed: List[str] = []
        if step_idx == 0:
            # Handle objects with zero burn_time that start on a burning cell.
            for obj in objects:
                if obj.burn_time <= 0 and fire_grid[obj.row][obj.col].is_burning:
                    if obj.obj_id not in destroyed:
                        destroyed.add(obj.obj_id)
                        newly_destroyed.append(obj.obj_id)
                        cumulative_loss += obj.value
        else:
            # Exposure update:
            # If an object's cell is burning for this tick, accumulate `dt`.
            # Destroy object once exposure reaches its burn_time threshold.
            for obj in objects:
                if obj.obj_id in destroyed:
                    continue
                if fire_grid[obj.row][obj.col].is_burning:
                    object_exposure[obj.obj_id] += dt
                    if object_exposure[obj.obj_id] + 1e-12 >= obj.burn_time:
                        destroyed.add(obj.obj_id)
                        newly_destroyed.append(obj.obj_id)
                        cumulative_loss += obj.value

        # Fire front is the set of currently active burning cells after updates.
        fire_front = [
            (r, c)
            for r in range(n)
            for c in range(m)
            if fire_grid[r][c].is_burning
        ]

        # Snapshot full grid state for playback/debugging (not a live reference).
        steps.append(
            SimulationStep(
                t=t,
                fire_front=fire_front,
                newly_ignited=newly_ignited,
                newly_destroyed=newly_destroyed,
                cumulative_loss=float(cumulative_loss),
                fire_grid=_clone_grid(fire_grid),
            )
        )

    return steps


def simulate_uniform(
    n: int,
    m: int,
    cell_burn_duration: float,
    max_time: float,
    dt: float,
    objects_at_cells: List[Dict[str, Any]],
) -> List[SimulationStep]:
    """Run expected simulation under a uniform prior over ignition source cells.

    Strategy:
    - Run `simulate_from_source` once for each possible source cell.
    - Aggregate burn/burnt/destroy statistics at each step.
    - Produce a representative thresholded grid plus explicit expected fields.
    """

    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")

    objects = _parse_objects(objects_at_cells, n, m)
    # Used to compute expected cumulative loss from object destruction probs.
    value_by_id = {obj.obj_id: obj.value for obj in objects}
    all_sources = [(r, c) for r in range(n) for c in range(m)]
    num_sources = len(all_sources)
    if num_sources == 0:
        return []

    # Run deterministic simulations for every possible source cell.
    all_runs: List[List[SimulationStep]] = []
    for source_r, source_c in all_sources:
        run_steps = simulate_from_source(
            n=n,
            m=m,
            cell_burn_duration=cell_burn_duration,
            max_time=max_time,
            dt=dt,
            source_r=source_r,
            source_c=source_c,
            objects_at_cells=objects_at_cells,
        )
        all_runs.append(run_steps)

    step_count = len(all_runs[0])

    # Per-step counters to later convert into probabilities by dividing by
    # `num_sources`.
    burning_counts = [[[0.0 for _ in range(m)] for _ in range(n)] for _ in range(step_count)]
    burnt_counts = [[[0.0 for _ in range(m)] for _ in range(n)] for _ in range(step_count)]
    destroyed_counts: List[Dict[str, float]] = [{obj.obj_id: 0.0 for obj in objects} for _ in range(step_count)]

    for run_steps in all_runs:
        # Within one run, once destroyed, an object stays destroyed forever.
        destroyed_so_far: set[str] = set()
        for i, step in enumerate(run_steps):
            for r in range(n):
                for c in range(m):
                    if step.fire_grid[r][c].is_burning:
                        burning_counts[i][r][c] += 1.0
                    if step.fire_grid[r][c].is_burnt_out:
                        burnt_counts[i][r][c] += 1.0

            destroyed_so_far.update(step.newly_destroyed)
            for obj_id in destroyed_so_far:
                if obj_id in destroyed_counts[i]:
                    destroyed_counts[i][obj_id] += 1.0

    aggregated_steps: List[SimulationStep] = []
    # For the derived thresholded representation, track previous state to
    # identify "newly ignited" and "newly destroyed" transitions.
    prev_threshold_burning = [[False for _ in range(m)] for _ in range(n)]
    prev_probs = {obj.obj_id: 0.0 for obj in objects}

    for i in range(step_count):
        expected_fire_grid = [
            [burning_counts[i][r][c] / num_sources for c in range(m)]
            for r in range(n)
        ]
        expected_burnt_grid = [
            [burnt_counts[i][r][c] / num_sources for c in range(m)]
            for r in range(n)
        ]

        # Representative boolean state for UI overlays. We still return explicit
        # probabilities in `expected_fire_grid`.
        threshold_burning = [
            [expected_fire_grid[r][c] >= 0.5 for c in range(m)]
            for r in range(n)
        ]

        fire_front: List[Tuple[int, int]] = []
        newly_ignited: List[Tuple[int, int]] = []
        fire_grid_snapshot: List[List[CellState]] = []

        for r in range(n):
            row_states: List[CellState] = []
            for c in range(m):
                is_burning = threshold_burning[r][c]
                is_burnt_out = (not is_burning) and (expected_burnt_grid[r][c] >= 0.5)
                row_states.append(CellState(is_burning=is_burning, is_burnt_out=is_burnt_out))

                if is_burning:
                    fire_front.append((r, c))
                if is_burning and not prev_threshold_burning[r][c]:
                    newly_ignited.append((r, c))
            fire_grid_snapshot.append(row_states)

        expected_loss_by_obj = {
            obj.obj_id: destroyed_counts[i][obj.obj_id] / num_sources for obj in objects
        }

        # Mark any object whose destruction probability increased this step.
        newly_destroyed = [
            obj_id
            for obj_id, prob in expected_loss_by_obj.items()
            if prob - prev_probs.get(obj_id, 0.0) > 1e-12
        ]

        # Expected cumulative loss is linear in object value * destruction prob.
        cumulative_loss = float(
            sum(value_by_id[obj_id] * expected_loss_by_obj[obj_id] for obj_id in expected_loss_by_obj)
        )

        t = all_runs[0][i].t
        aggregated_steps.append(
            SimulationStep(
                t=t,
                fire_front=fire_front,
                newly_ignited=newly_ignited,
                newly_destroyed=newly_destroyed,
                cumulative_loss=cumulative_loss,
                fire_grid=fire_grid_snapshot,
                expected_fire_grid=expected_fire_grid,
                expected_loss_by_obj=expected_loss_by_obj,
            )
        )

        prev_threshold_burning = threshold_burning
        prev_probs = expected_loss_by_obj

    return aggregated_steps
