from __future__ import annotations

"""Branch-and-bound placement solver for expected fire loss.

Loss model (agreed formula)
---------------------------
For a single fire source s and object k placed at cell p_k:

    L(k, s) = v_k   if  manhattan(s, p_k) <= t_max - b_k
              0      otherwise

Assuming the fire source is drawn uniformly at random from all n*m cells,
the expected loss for object k placed at cell p_k is:

    E[L(k)] = v_k * |{s : manhattan(s, p_k) <= t_max - b_k}| / (n * m)
            = v_k * reachable_sources(p_k, t_max - b_k) / (n * m)

Where reachable_sources(p, d) counts grid cells within Manhattan distance d
of p.  If t_max - b_k < 0 the object can never be destroyed (deadline passed
before fire could arrive), so its contribution is 0.

Total expected loss = sum over all objects of E[L(k)].

The optimizer minimises this quantity over all injective assignments of
objects to cells using branch-and-bound with an optimistic lower bound.

Backward compatibility
----------------------
The public API is unchanged:
    solve_global_expected_loss(n, m, objects, t_max=10) -> dict

`t_max` defaults to 10 so existing callers that omit it continue to work.
`flammability` is accepted on each object (and stored) but is not used in
the loss formula, matching the agreed model.  Removing it from the struct
would break callers that pass it in.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes (public interface unchanged)
# ---------------------------------------------------------------------------

@dataclass
class BBObject:
    """Input object record used by the optimizer."""
    obj_id: str
    value: float
    burn_time: float
    flammability: float   # retained for API compatibility; not used in loss


@dataclass
class AssignmentEntry:
    """One optimised placement decision for a single object."""
    obj_id: str
    row: int
    col: int
    expected_contribution: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reachable_count(row: int, col: int, d: int, n: int, m: int) -> int:
    """Count grid cells within Manhattan distance d of (row, col).

    Row-sweep approach: for each row offset dr in [-d, d], the valid column
    range is [col - (d-|dr|), col + (d-|dr|)], clipped to [0, m-1].
    Runs in O(d) which is fast enough for the t_max values used in practice.
    """
    if d < 0:
        return 0
    count = 0
    for dr in range(-d, d + 1):
        r = row + dr
        if r < 0 or r >= n:
            continue
        dc_max = d - abs(dr)
        c_lo = max(0, col - dc_max)
        c_hi = min(m - 1, col + dc_max)
        count += c_hi - c_lo + 1
    return count


def _expected_loss_coeff(row: int, col: int, obj: BBObject,
                         t_max: int, n: int, m: int) -> float:
    """Expected loss for `obj` placed at (row, col) under a uniform source.

    E[L(k)] = v_k * reachable_sources(p_k, t_max - b_k) / (n * m)

    Returns 0.0 if the deadline t_max - b_k is negative (object is immune).
    """
    deadline = t_max - obj.burn_time
    if deadline < 0:
        return 0.0
    reachable = _reachable_count(row, col, int(deadline), n, m)
    return obj.value * reachable / (n * m)


def _precompute_loss_table(
    objects: List[BBObject],
    n: int,
    m: int,
    t_max: int,
) -> Tuple[List[List[float]], List[int]]:
    """Build a (num_objects x num_cells) table of expected-loss values.

    Also returns the sorted object order (highest max-possible-loss first)
    for better B&B pruning.

    Table layout: loss_table[obj_sorted_idx][cell_idx]
    """
    num_cells = n * m

    # For each object, compute its expected loss at every cell.
    raw: List[List[float]] = []
    max_possible: List[float] = []
    for obj in objects:
        row_losses: List[float] = []
        for cell_idx in range(num_cells):
            r, c = divmod(cell_idx, m)
            row_losses.append(_expected_loss_coeff(r, c, obj, t_max, n, m))
        raw.append(row_losses)
        max_possible.append(max(row_losses))

    # Sort objects by their maximum achievable loss (desc) so the B&B
    # branching order focuses on the most impactful objects first.
    sorted_indices = sorted(range(len(objects)),
                            key=lambda i: max_possible[i], reverse=True)

    loss_table = [raw[i] for i in sorted_indices]
    return loss_table, sorted_indices


def _parse_objects(objects: Sequence[Dict[str, Any]]) -> List[BBObject]:
    """Normalise raw JSON-like objects into typed optimiser records."""
    parsed: List[BBObject] = []
    for idx, obj in enumerate(objects):
        parsed.append(BBObject(
            obj_id=str(obj.get("obj_id", f"obj_{idx + 1}")),
            value=float(obj["value"]),
            burn_time=max(0.0, float(obj.get("burn_time", 1.0))),
            flammability=max(0.0, float(obj.get("flammability", 1.0))),
        ))
    return parsed


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_global_expected_loss(
    n: int,
    m: int,
    objects: Sequence[Dict[str, Any]],
    t_max: int = 10,                   # NEW — defaults to 10 for back-compat
) -> Dict[str, Any]:
    """Solve object-to-cell assignment minimising expected loss.

    Uses the agreed formula:
        E[L(k)] = v_k * |sources that can reach p_k within t_max - b_k steps|
                        / (n * m)

    Parameters
    ----------
    n, m    : grid dimensions
    objects : list of dicts with keys obj_id, value, burn_time, flammability
    t_max   : maximum simulation time (fire travel budget).  Defaults to 10.

    Returns
    -------
    {
        "best_loss"  : float,
        "assignment" : [{"obj_id", "row", "col", "expected_contribution"}, ...]
    }
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")

    parsed = _parse_objects(objects)
    if len(parsed) > n * m:
        raise ValueError("More objects than cells: cannot place all objects")

    num_cells = n * m
    cell_indices = list(range(num_cells))

    # loss_table[sorted_obj_idx][cell_idx] = E[L] for that (obj, cell) pair.
    # sorted_obj_order maps sorted position -> original parsed index.
    loss_table, sorted_obj_order = _precompute_loss_table(parsed, n, m, t_max)
    sorted_objects = [parsed[i] for i in sorted_obj_order]
    num_objects = len(sorted_objects)

    best_loss: float = float("inf")
    best_assignment: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Optimistic lower bound
    # ------------------------------------------------------------------
    def optimistic_lower_bound(start_idx: int,
                                remaining_cell_ids: List[int]) -> float:
        """Lower bound on remaining loss.

        For each remaining object (in sorted order) pair it with its
        minimum-loss cell among those still available.  This is optimistic
        because each object gets an independent best cell (they may overlap),
        so the true loss can only be >= this bound.
        """
        bound = 0.0
        for obj_i in range(start_idx, num_objects):
            # Minimum expected loss this object can achieve in any remaining cell.
            min_loss = min(loss_table[obj_i][c] for c in remaining_cell_ids)
            bound += min_loss
        return bound

    # ------------------------------------------------------------------
    # DFS
    # ------------------------------------------------------------------
    def dfs(obj_idx: int,
            remaining_cell_ids: List[int],
            current_loss: float,
            assignment: Dict[str, int]) -> None:
        nonlocal best_loss, best_assignment

        # Leaf: all objects placed.
        if obj_idx >= num_objects:
            if current_loss < best_loss:
                best_loss = current_loss
                best_assignment = dict(assignment)
            return

        # Prune if optimistic future cannot beat incumbent.
        lb = current_loss + optimistic_lower_bound(obj_idx, remaining_cell_ids)
        if lb >= best_loss:
            return

        obj = sorted_objects[obj_idx]

        # Try cells in ascending expected-loss order for this object so that
        # low-loss assignments are explored first, tightening the bound fast.
        candidate_cells = sorted(remaining_cell_ids,
                                 key=lambda c: loss_table[obj_idx][c])

        for cell_idx in candidate_cells:
            contrib = loss_table[obj_idx][cell_idx]
            assignment[obj.obj_id] = cell_idx
            next_cells = [c for c in remaining_cell_ids if c != cell_idx]
            dfs(obj_idx + 1, next_cells,
                current_loss + contrib, assignment)
            assignment.pop(obj.obj_id, None)

    dfs(0, cell_indices, 0.0, {})

    if best_assignment is None:
        return {"best_loss": 0.0, "assignment": []}

    # Reconstruct in original input order for stable visualiser output.
    assignment_entries: List[AssignmentEntry] = []
    for obj in parsed:
        cell_idx = best_assignment[obj.obj_id]
        r, c = divmod(cell_idx, m)
        contrib = _expected_loss_coeff(r, c, obj, t_max, n, m)
        assignment_entries.append(AssignmentEntry(
            obj_id=obj.obj_id,
            row=r,
            col=c,
            expected_contribution=float(contrib),
        ))

    return {
        "best_loss": float(best_loss),
        "assignment": [e.__dict__ for e in assignment_entries],
    }
