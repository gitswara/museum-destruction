from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class CellState:
    is_burning: bool = False
    is_burnt_out: bool = False


@dataclass
class PlacedObject:
    obj_id: str
    row: int
    col: int
    value: float
    burn_time: float


@dataclass
class SimulationStep:
    t: float
    fire_front: List[Tuple[int, int]]
    newly_ignited: List[Tuple[int, int]]
    newly_destroyed: List[str]
    cumulative_loss: float
    fire_grid: List[List[CellState]]
    expected_fire_grid: Optional[List[List[float]]] = None
    expected_loss_by_obj: Optional[Dict[str, float]] = None


def _neighbor_cells(r: int, c: int, n: int, m: int) -> Iterable[Tuple[int, int]]:
    if r > 0:
        yield (r - 1, c)
    if r < n - 1:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c < m - 1:
        yield (r, c + 1)


def _clone_grid(grid: List[List[CellState]]) -> List[List[CellState]]:
    return [[CellState(cell.is_burning, cell.is_burnt_out) for cell in row] for row in grid]


def _parse_objects(objects_at_cells: List[Dict[str, Any]], n: int, m: int) -> List[PlacedObject]:
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

    fire_grid = [[CellState() for _ in range(m)] for _ in range(n)]
    ignite_time: Dict[Tuple[int, int], float] = {}
    object_exposure: Dict[str, float] = {obj.obj_id: 0.0 for obj in objects}
    destroyed: set[str] = set()
    cumulative_loss = 0.0

    def ignite_cell(r: int, c: int, t: float) -> bool:
        cell = fire_grid[r][c]
        if cell.is_burning or cell.is_burnt_out:
            return False
        cell.is_burning = True
        ignite_time[(r, c)] = t
        return True

    initial_new_ignitions: List[Tuple[int, int]] = []
    if ignite_cell(source_r, source_c, 0.0):
        initial_new_ignitions.append((source_r, source_c))

    times: List[float] = []
    t_cursor = 0.0
    while t_cursor <= max_time + 1e-12:
        times.append(round(t_cursor, 10))
        t_cursor += dt

    steps: List[SimulationStep] = []

    for step_idx, t in enumerate(times):

        if step_idx == 0:
            newly_ignited = initial_new_ignitions
        else:
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

        # Burnout update at time t.
        for (r, c), start_t in list(ignite_time.items()):
            if fire_grid[r][c].is_burning and (t - start_t) >= cell_burn_duration:
                fire_grid[r][c].is_burning = False
                fire_grid[r][c].is_burnt_out = True

        newly_destroyed: List[str] = []
        if step_idx == 0:
            # Handle instantly destroyed objects.
            for obj in objects:
                if obj.burn_time <= 0 and fire_grid[obj.row][obj.col].is_burning:
                    if obj.obj_id not in destroyed:
                        destroyed.add(obj.obj_id)
                        newly_destroyed.append(obj.obj_id)
                        cumulative_loss += obj.value
        else:
            # Approximate exposure accumulation over this interval by current burning state.
            for obj in objects:
                if obj.obj_id in destroyed:
                    continue
                if fire_grid[obj.row][obj.col].is_burning:
                    object_exposure[obj.obj_id] += dt
                    if object_exposure[obj.obj_id] + 1e-12 >= obj.burn_time:
                        destroyed.add(obj.obj_id)
                        newly_destroyed.append(obj.obj_id)
                        cumulative_loss += obj.value

        fire_front = [
            (r, c)
            for r in range(n)
            for c in range(m)
            if fire_grid[r][c].is_burning
        ]

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
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")

    objects = _parse_objects(objects_at_cells, n, m)
    value_by_id = {obj.obj_id: obj.value for obj in objects}
    all_sources = [(r, c) for r in range(n) for c in range(m)]
    num_sources = len(all_sources)
    if num_sources == 0:
        return []

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

    burning_counts = [[[0.0 for _ in range(m)] for _ in range(n)] for _ in range(step_count)]
    burnt_counts = [[[0.0 for _ in range(m)] for _ in range(n)] for _ in range(step_count)]
    destroyed_counts: List[Dict[str, float]] = [{obj.obj_id: 0.0 for obj in objects} for _ in range(step_count)]

    for run_steps in all_runs:
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

        newly_destroyed = [
            obj_id
            for obj_id, prob in expected_loss_by_obj.items()
            if prob - prev_probs.get(obj_id, 0.0) > 1e-12
        ]

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
