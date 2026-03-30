from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory

from branch_and_bound import solve_global_expected_loss
from fire_simulator import SimulationStep, simulate_from_source, simulate_uniform


app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def _as_int(payload: Dict[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"Missing key: {key}")
    return int(payload[key])


def _as_float(payload: Dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"Missing key: {key}")
    return float(payload[key])


def _serialize_step(step: SimulationStep) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "t": float(step.t),
        "fire_front": [[int(r), int(c)] for r, c in step.fire_front],
        "newly_ignited": [[int(r), int(c)] for r, c in step.newly_ignited],
        "newly_destroyed": [str(obj_id) for obj_id in step.newly_destroyed],
        "cumulative_loss": float(step.cumulative_loss),
        "fire_grid": [
            [
                {
                    "is_burning": bool(cell.is_burning),
                    "is_burnt_out": bool(cell.is_burnt_out),
                }
                for cell in row
            ]
            for row in step.fire_grid
        ],
    }

    if step.expected_fire_grid is not None:
        out["expected_fire_grid"] = [
            [float(value) for value in row] for row in step.expected_fire_grid
        ]

    if step.expected_loss_by_obj is not None:
        out["expected_loss_by_obj"] = {
            str(obj_id): float(prob)
            for obj_id, prob in step.expected_loss_by_obj.items()
        }

    return out


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.route("/solve", methods=["POST", "OPTIONS"])
def solve_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")

        n = _as_int(payload, "n")
        m = _as_int(payload, "m")
        objects = payload.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("objects must be a list")

        result = solve_global_expected_loss(n=n, m=m, objects=objects)
        return jsonify(result)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to solve placement: {exc}"}), 500


@app.route("/simulate", methods=["POST", "OPTIONS"])
def simulate_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")

        n = _as_int(payload, "n")
        m = _as_int(payload, "m")
        cell_burn_duration = _as_float(payload, "cell_burn_duration")
        max_time = _as_float(payload, "max_time")
        dt = _as_float(payload, "dt")
        source_r = _as_int(payload, "source_r")
        source_c = _as_int(payload, "source_c")
        objects_at_cells = payload.get("objects_at_cells", [])

        if not isinstance(objects_at_cells, list):
            raise ValueError("objects_at_cells must be a list")

        steps = simulate_from_source(
            n=n,
            m=m,
            cell_burn_duration=cell_burn_duration,
            max_time=max_time,
            dt=dt,
            source_r=source_r,
            source_c=source_c,
            objects_at_cells=objects_at_cells,
        )

        return jsonify({"steps": [_serialize_step(step) for step in steps]})

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to run simulation: {exc}"}), 500


@app.route("/simulate_uniform", methods=["POST", "OPTIONS"])
def simulate_uniform_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")

        n = _as_int(payload, "n")
        m = _as_int(payload, "m")
        cell_burn_duration = _as_float(payload, "cell_burn_duration")
        max_time = _as_float(payload, "max_time")
        dt = _as_float(payload, "dt")
        objects_at_cells = payload.get("objects_at_cells", [])

        if not isinstance(objects_at_cells, list):
            raise ValueError("objects_at_cells must be a list")

        steps = simulate_uniform(
            n=n,
            m=m,
            cell_burn_duration=cell_burn_duration,
            max_time=max_time,
            dt=dt,
            objects_at_cells=objects_at_cells,
        )

        return jsonify({"steps": [_serialize_step(step) for step in steps]})

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to run uniform simulation: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
