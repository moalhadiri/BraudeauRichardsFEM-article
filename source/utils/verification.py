"""Short numerical checks used by the reviewer examples."""
import json
from pathlib import Path
import numpy as np
from ..api import build_validation


def record(name, passed, **values):
    return dict(name=name, status="passed" if passed else "failed", **values)


def numerical_checks():
    checks = []
    rng = np.random.default_rng(20261024)
    for dimension in (2, 3):
        for scheme in ("implicit", "semi_implicit"):
            name = f"{dimension}D {scheme} Jacobian"
            data = other = None
            try:
                data = build_validation(dimension, scheme, nx=5, workers=1)
                op, p, free = data["operator"], data["p"], data["interior"]
                previous = -.2 + .02 * np.sin(np.pi * p).prod(axis=1)
                state = previous + .01 * np.cos(p.sum(axis=1))
                direction = np.zeros(len(p))
                direction[free] = rng.normal(size=len(free))
                direction /= np.linalg.norm(direction)
                force = np.zeros(len(p))
                action = (op.jacobien(state, previous, .05) @ direction)[free]
                errors = []
                for epsilon in (1e-4, 5e-5, 2.5e-5):
                    difference = (op.residu(state + epsilon * direction, previous, .05, force) - op.residu(state - epsilon * direction, previous, .05, force)) / (2 * epsilon)
                    errors.append(float(np.linalg.norm(difference[free] - action) / max(np.linalg.norm(action), 1e-300)))
                checks.append(record(name, min(errors) < 2e-7, value=min(errors), tolerance=2e-7, message="Centred directional differences at three perturbation sizes."))
                fields = data["laws"].fields(state)
                grad = np.einsum("ej,ejk->ek", state[op.t], op.G)
                terms = (fields["k"].v * fields["g1"].d, fields["ell"].v * fields["g2"].d, fields["g2"].v * fields["ell"].d)
                active = all(np.max(np.abs(value)) > 1e-8 for value in terms) and np.max(abs(grad[:, -1])) > 1e-8
                checks.append(record(f"{dimension}D {scheme} source terms active", active, message="The probe activates both quadratic-gradient coefficients and the gravity coefficient."))
                other = build_validation(dimension, scheme, nx=5, workers=2)
                r1, r2 = op.residu(state, previous, .05, force), other["operator"].residu(state, previous, .05, force)
                a1, a2 = op.jacobien(state, previous, .05).toarray(), other["operator"].jacobien(state, previous, .05).toarray()
                identical = np.array_equal(r1, r2) and np.array_equal(a1, a2)
                checks.append(record(f"{dimension}D {scheme} one/two assembly workers", identical, value=float(max(np.max(abs(r1-r2)), np.max(abs(a1-a2)))), tolerance=0., message="Bitwise residual and Jacobian comparison on the same mesh."))
            except Exception as error:
                checks.append(record(name, False, message=f"{type(error).__name__}: {error}"))
            finally:
                if data is not None:
                    data["operator"].close()
                if other is not None:
                    other["operator"].close()
    return checks


def case_checks(result, paper_data):
    case_id = result["id"]
    checks = []
    final, p = result["final"], result["p"]
    boundary = result["boundary"]
    if result["problem"] == "wet_slotted_sphere":
        trace = np.full(len(boundary), -5.)
    elif result["problem"] == "dry_infiltration":
        points = p[boundary]
        end = np.where(np.isclose(points[:, 0], 0.), 3., 3. * (1. - points[:, 1]))
        trace = -400. + min(result["t_final"] / 20., 1.) * (end + 400.)
    else:
        trace = np.zeros(len(boundary))
    boundary_error = float(np.max(abs(final[boundary] - trace)))
    checks.append(record(case_id + " boundary", boundary_error <= 1e-10, value=boundary_error, tolerance=1e-10))
    checks.append(record(case_id + " finite field", bool(np.isfinite(final).all())))
    final_correction = max((step["final_relative_correction"] for step in result["accepted_steps"]), default=float("inf"))
    checks.append(record(case_id + " Newton stopping", final_correction <= 1e-8, value=final_correction, tolerance=1e-8))
    residuals = [row["linear_relative_residual"] for row in result["correction_rows"] if row.get("accepted", True)]
    maximum = max(residuals, default=float("inf"))
    if result["linear_solver"] == "GMRES-ILU":
        checks.append(record(case_id + " true GMRES residual", maximum <= 1.05e-10, value=maximum, tolerance=1.05e-10))
    result["boundary_error"] = boundary_error
    result["maximum_linear_relative_residual"] = maximum
    if result.get("problem") != "polynomial_validation" or result.get("time_model") != "stationary":
        return checks
    references = json.loads((Path(paper_data) / "published_spatial_cases.json").read_text(encoding="utf8"))["cases"]
    for reference in references:
        same = (reference["dimension"] == result["dimension"] and reference["scheme"].replace("-", "_") == result["scheme"]
                and reference["nodes_per_direction"] == result["nx"] and reference["tau"] == result["dt"]
                and reference["final_time"] == result["t_final"] and reference["relaxation"] == result["relaxation"])
        if not same:
            continue
        for norm, key in (("L2", "error_L2"), ("H1", "error_H1")):
            value = result[key]
            target = reference[norm]
            tolerance = max(1.01 * target["rounding_half_unit"], 1e-12)
            checks.append(record(f"{case_id} published {norm}", abs(value - target["value"]) <= tolerance,
                                 value=value, reference=target["value"], tolerance=tolerance, message=f"Paper table {reference['paper_table']}; allowance for the printed last digit."))
        break
    return checks
