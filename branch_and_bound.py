from __future__ import annotations

"""Branch-and-bound placement solver for expected fire loss.

This module provides a compact optimization model used by the app's
"Find optimal placement (B&B)" feature.

High-level idea:
- Each grid cell is assigned a heuristic risk score (more central cells are
  treated as riskier in this simplified model).
- Each object gets an impact coefficient based on value, flammability, and
  burn resistance (`burn_time`).
- We minimize total expected loss contribution by assigning objects to distinct
  cells.
- Branch-and-bound uses an optimistic lower bound to prune unpromising search
  branches early.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class BBObject:
    """Input object record used by the optimizer."""

    obj_id: str
    value: float
    burn_time: float
    flammability: float


@dataclass
class AssignmentEntry:
    """One optimized placement decision for a single object."""

    obj_id: str
    row: int
    col: int
    expected_contribution: float


def _cell_risk_map(n: int, m: int) -> List[Tuple[int, int, float]]:
    """Return `(row, col, risk)` for every cell in the `n x m` grid.

    Risk heuristic:
    - Compute each cell's average Manhattan distance to all cells.
    - Convert that into a centrality score in [0, 1].
    - Map to risk in [0.2, 1.0] so no cell has zero risk.
    """

    # Cells closer (in average Manhattan distance) to all other cells are
    # treated as riskier (more "central" means easier spread reachability).
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
    """Normalize raw JSON-like objects into typed optimizer records."""

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
    """Solve object-to-cell assignment minimizing modeled expected loss.

    Returns:
    - `best_loss`: total minimized expected loss (heuristic model).
    - `assignment`: list of `{obj_id, row, col, expected_contribution}`.
    """

    # Basic domain checks keep API error messages explicit.
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")

    parsed = _parse_objects(objects)
    if len(parsed) > n * m:
        raise ValueError("More objects than cells: cannot place all objects")

    # Precompute cell risks and index-addressable structures for fast recursion.
    cells = _cell_risk_map(n, m)
    cell_indices = list(range(len(cells)))
    risks = [risk for _, _, risk in cells]

    # Lightweight expected-loss coefficient:
    #   coeff = value * flammability / (1 + burn_time)
    # Higher value/flammability increases risk; higher burn_time lowers it.
    coeff_by_obj: Dict[str, float] = {}
    for obj in parsed:
        resistance = 1.0 + obj.burn_time
        coeff_by_obj[obj.obj_id] = obj.value * obj.flammability / resistance

    # Branch on highest-impact objects first. Better ordering often yields
    # stronger pruning earlier in branch-and-bound.
    sorted_objects = sorted(parsed, key=lambda x: coeff_by_obj[x.obj_id], reverse=True)

    best_loss = float("inf")
    best_assignment: Optional[Dict[str, int]] = None

    def optimistic_lower_bound(start_idx: int, remaining_cell_ids: List[int]) -> float:
        """Lower bound on additional loss for remaining objects.

        Constructed by pairing largest remaining object coefficients with
        smallest remaining cell risks. This is optimistic (best-case), so if
        current loss + this bound is already worse than incumbent, prune branch.
        """

        remaining_objs = sorted_objects[start_idx:]
        if not remaining_objs:
            return 0.0

        remaining_coeffs = sorted((coeff_by_obj[o.obj_id] for o in remaining_objs), reverse=True)
        remaining_risks = sorted((risks[i] for i in remaining_cell_ids))

        k = min(len(remaining_coeffs), len(remaining_risks))
        return sum(remaining_coeffs[i] * remaining_risks[i] for i in range(k))

    def dfs(obj_idx: int, remaining_cell_ids: List[int], current_loss: float, assignment: Dict[str, int]) -> None:
        """Depth-first branch-and-bound search over assignments."""

        nonlocal best_loss, best_assignment

        # Leaf: all objects assigned.
        if obj_idx >= len(sorted_objects):
            if current_loss < best_loss:
                best_loss = current_loss
                best_assignment = dict(assignment)
            return

        # Prune if even optimistic continuation cannot beat current best.
        lb = current_loss + optimistic_lower_bound(obj_idx, remaining_cell_ids)
        if lb >= best_loss:
            return

        obj = sorted_objects[obj_idx]
        coeff = coeff_by_obj[obj.obj_id]

        # Try safer cells first (lower risk) to quickly discover stronger
        # incumbents, which then improves later pruning.
        candidate_cells = sorted(remaining_cell_ids, key=lambda i: risks[i])
        for cell_idx in candidate_cells:
            risk = risks[cell_idx]
            assignment[obj.obj_id] = cell_idx
            next_cells = [i for i in remaining_cell_ids if i != cell_idx]
            dfs(obj_idx + 1, next_cells, current_loss + coeff * risk, assignment)
            assignment.pop(obj.obj_id, None)

    # Solve from root state: no objects assigned yet.
    dfs(0, cell_indices, 0.0, {})

    if best_assignment is None:
        return {"best_loss": 0.0, "assignment": []}

    # Reconstruct output in original input object order for predictable UX.
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
