from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class BBObject:
    obj_id: str
    value: float
    burn_time: float
    flammability: float


@dataclass
class AssignmentEntry:
    obj_id: str
    row: int
    col: int
    expected_contribution: float


def _cell_risk_map(n: int, m: int) -> List[Tuple[int, int, float]]:
    # Cells closer (in average Manhattan distance) to all other cells are treated as riskier.
    distances: List[Tuple[int, int, float]] = []
    all_cells = [(r, c) for r in range(n) for c in range(m)]

    for r, c in all_cells:
        total = 0.0
        for rr, cc in all_cells:
            total += abs(r - rr) + abs(c - cc)
        avg_distance = total / len(all_cells)
        distances.append((r, c, avg_distance))

    min_d = min(d for _, _, d in distances)
    max_d = max(d for _, _, d in distances)

    out: List[Tuple[int, int, float]] = []
    for r, c, avg_d in distances:
        if max_d - min_d < 1e-12:
            centrality = 0.5
        else:
            centrality = 1.0 - ((avg_d - min_d) / (max_d - min_d))

        # Keep risk bounded away from 0 to avoid degenerate optimization.
        risk = 0.2 + 0.8 * centrality
        out.append((r, c, risk))

    return out


def _parse_objects(objects: Sequence[Dict[str, Any]]) -> List[BBObject]:
    parsed: List[BBObject] = []
    for idx, obj in enumerate(objects):
        parsed.append(
            BBObject(
                obj_id=str(obj.get("obj_id", f"obj_{idx + 1}")),
                value=float(obj["value"]),
                burn_time=max(0.0, float(obj.get("burn_time", 1.0))),
                flammability=max(0.0, float(obj.get("flammability", 1.0))),
            )
        )
    return parsed


def solve_global_expected_loss(n: int, m: int, objects: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")

    parsed = _parse_objects(objects)
    if len(parsed) > n * m:
        raise ValueError("More objects than cells: cannot place all objects")

    cells = _cell_risk_map(n, m)
    cell_indices = list(range(len(cells)))
    risks = [risk for _, _, risk in cells]

    # A lightweight expected-loss model combining object value, flammability, and resistance.
    coeff_by_obj: Dict[str, float] = {}
    for obj in parsed:
        resistance = 1.0 + obj.burn_time
        coeff_by_obj[obj.obj_id] = obj.value * obj.flammability / resistance

    # Branch on high-impact objects first for stronger pruning.
    sorted_objects = sorted(parsed, key=lambda x: coeff_by_obj[x.obj_id], reverse=True)

    best_loss = float("inf")
    best_assignment: Optional[Dict[str, int]] = None

    def optimistic_lower_bound(start_idx: int, remaining_cell_ids: List[int]) -> float:
        remaining_objs = sorted_objects[start_idx:]
        if not remaining_objs:
            return 0.0

        remaining_coeffs = sorted((coeff_by_obj[o.obj_id] for o in remaining_objs), reverse=True)
        remaining_risks = sorted((risks[i] for i in remaining_cell_ids))

        k = min(len(remaining_coeffs), len(remaining_risks))
        return sum(remaining_coeffs[i] * remaining_risks[i] for i in range(k))

    def dfs(obj_idx: int, remaining_cell_ids: List[int], current_loss: float, assignment: Dict[str, int]) -> None:
        nonlocal best_loss, best_assignment

        if obj_idx >= len(sorted_objects):
            if current_loss < best_loss:
                best_loss = current_loss
                best_assignment = dict(assignment)
            return

        lb = current_loss + optimistic_lower_bound(obj_idx, remaining_cell_ids)
        if lb >= best_loss:
            return

        obj = sorted_objects[obj_idx]
        coeff = coeff_by_obj[obj.obj_id]

        # Try safer cells first (lower risk) to quickly find strong incumbents.
        candidate_cells = sorted(remaining_cell_ids, key=lambda i: risks[i])
        for cell_idx in candidate_cells:
            risk = risks[cell_idx]
            assignment[obj.obj_id] = cell_idx
            next_cells = [i for i in remaining_cell_ids if i != cell_idx]
            dfs(obj_idx + 1, next_cells, current_loss + coeff * risk, assignment)
            assignment.pop(obj.obj_id, None)

    dfs(0, cell_indices, 0.0, {})

    if best_assignment is None:
        return {"best_loss": 0.0, "assignment": []}

    assignment_entries: List[AssignmentEntry] = []
    for obj in parsed:
        cell_idx = best_assignment[obj.obj_id]
        r, c, risk = cells[cell_idx]
        expected_contribution = coeff_by_obj[obj.obj_id] * risk
        assignment_entries.append(
            AssignmentEntry(
                obj_id=obj.obj_id,
                row=r,
                col=c,
                expected_contribution=float(expected_contribution),
            )
        )

    return {
        "best_loss": float(best_loss),
        "assignment": [entry.__dict__ for entry in assignment_entries],
    }
