"""Small entry points for the validation and physical cases in the article."""
from functools import partial
import math
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .LocalGlobalMapping.mesh3d import kpde3dumsh
from .MatrixAssembly.mass3d import kpde3dmass
from .MatrixAssembly.triangles import mesh, mass_matrix, OperateurSemiImplicite2D
from .MatrixAssembly.implicit2d import OperateurImplicite2D
from .MatrixAssembly.parallel3d import OperateurNewtonParallele
from .MatrixAssembly.semi3d import OperateurSemiImpliciteParallele
from .MatrixAssembly.physical2d import rectangle_mesh, PhysicalImplicit2D
from .models import validation2d, validation3d, fourier3d
from .models.smooth import Implicit3DLaws
from .models.physical import PhysicalLaws, initial as physical_initial
from .solvers.newton import newton_step, NewtonFailure, NEWTON_TOLERANCE, MAX_NEWTON_ITERATIONS


def _mesh3d(nx, length=1.):
    p, t, *_ = kpde3dumsh(0., 1., 0., 1., 0., 1., nx, nx, nx)
    boundary = np.flatnonzero(np.any(np.isclose(p, 0.) | np.isclose(p, 1.), axis=1))
    interior = np.setdiff1d(np.arange(len(p)), boundary)
    return p * length, t, boundary, interior


def _check_grid(nx, workers):
    if isinstance(nx, bool) or int(nx) != nx or nx < 3:
        raise ValueError("nx must be an integer of at least three")
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")


def build_validation(dimension=2, scheme="implicit", nx=9, time_model="stationary", workers=1):
    """Return native operators and data; the caller closes the operator afterwards."""
    _check_grid(nx, workers)
    if scheme not in ("implicit", "semi_implicit") or dimension not in (2, 3):
        raise ValueError("Use implicit or semi_implicit in dimension two or three")
    allowed = ("stationary", "cubic") if dimension == 2 else ("stationary", "exponential")
    if time_model not in allowed:
        raise ValueError(f"time_model must be one of {allowed}")
    laws = Implicit3DLaws()
    if dimension == 2:
        p, t, boundary, interior = mesh(nx)
        mass = mass_matrix(p, t)
        cls = OperateurImplicite2D if scheme == "implicit" else OperateurSemiImplicite2D
        model = validation2d
    else:
        p, t, boundary, interior = _mesh3d(nx)
        mass = kpde3dmass(p, t, 1.)
        cls = OperateurNewtonParallele if scheme == "implicit" else OperateurSemiImpliciteParallele
        model = validation3d
    operator = cls(p, t, laws, forme="psi", gravite="volume", workers=workers)
    return dict(operator=operator, mass=mass, laws=laws, p=p, t=t, boundary=boundary, interior=interior,
                exact=partial(model.exact, time_model=time_model), forcing=partial(model.forcing, laws=laws, time_model=time_model),
                errors=partial(model.errors, time_model=time_model))


def build_physical_3d(dataset="Braudeau", nx=9, workers=1):
    """The slot is part of the initial pressure, on a cube of side 2.75 cm."""
    _check_grid(nx, workers)
    p, t, boundary, interior = _mesh3d(nx, length=2.75)
    laws = PhysicalLaws(dataset)
    initial = physical_initial(p / 2.75)
    initial[boundary] = -5.
    mass = kpde3dmass(p, t, 1.)
    operator = OperateurNewtonParallele(p, t, laws, forme="psi", gravite="volume", workers=workers)
    return dict(operator=operator, mass=mass, laws=laws, p=p, t=t, boundary=boundary, interior=interior, initial=initial)


def build_physical_2d(nx=21, nz=11, workers=1):
    """Only the lateral edges prescribe pressure; horizontal edges have zero total flux."""
    _check_grid(nx, workers)
    if isinstance(nz, bool) or int(nz) != nz or nz < 3:
        raise ValueError("nz must be an integer of at least three")
    p, t, boundary, interior = rectangle_mesh(nx, nz, Lx=2., Lz=1.)
    laws = PhysicalLaws("Braudeau")
    mass = mass_matrix(p, t)
    operator = PhysicalImplicit2D(p, t, laws, forme="psi", gravite="volume", workers=workers)
    return dict(operator=operator, mass=mass, laws=laws, p=p, t=t, boundary=boundary, interior=interior,
                initial=np.full(len(p), -400.))


def _summary(data, initial, final, counts, rows, steps, elapsed, **metadata):
    return dict(status="complete", run_status="complete", p=data["p"], t=data["t"], initial=initial, final=final,
                boundary=data["boundary"], interior=data["interior"], nodes=len(data["p"]), elements=len(data["t"]),
                newton_iterations=counts, newton_iterations_max=max(counts, default=0), newton_iterations_total=sum(counts),
                newton_iterations_mean=float(np.mean(counts)) if counts else 0., correction_rows=rows, accepted_steps=steps,
                completed_steps=len(steps), reached_time=steps[-1]["time"] if steps else 0.,
                elapsed_seconds=elapsed, newton_tolerance=NEWTON_TOLERANCE, maximum_newton_iterations=MAX_NEWTON_ITERATIONS,
                pressure_min=float(final.min()), pressure_max=float(final.max()),
                temporal_storage="lumped pressure difference", gravity="volume", **metadata)


def _fixed_run(data, dt, t_final, relaxation, scheme, **metadata):
    if not np.isfinite([dt, t_final, relaxation]).all() or dt <= 0 or t_final <= 0:
        raise ValueError("Times and relaxation must be finite, with positive times")
    count = round(t_final / dt)
    if not math.isclose(count * dt, t_final, rel_tol=0, abs_tol=1e-12):
        raise ValueError("t_final must be an integer multiple of dt")
    p = data["p"]
    boundary, interior, operator, mass = (data[key] for key in ("boundary", "interior", "operator", "mass"))
    physical = "initial" in data
    initial = data["initial"].copy() if physical else data["exact"](p, 0.)[0]
    previous = initial.copy()
    counts, rows, steps = [], [], []
    saved_states = {"0": initial.copy()}
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            for step in range(1, count + 1):
                target = round(step * dt, 12)
                source_time = round((step - 1) * dt, 12) if scheme == "semi_implicit" else target
                force = np.zeros(len(p)) if physical else data["forcing"](p, source_time)
                trace = np.full(len(boundary), -5.) if physical else np.zeros(len(boundary))
                state, history, seconds = newton_step(operator, previous, dt, force, boundary, interior, trace, mass, relaxation)
                rows.extend(dict(step=step, time=target, dt=dt, **row) for row in history)
                residual = operator.residu(state, previous, dt, force)[interior]
                steps.append(dict(step=step, time=target, dt=dt, newton_iterations=len(history),
                                  final_relative_correction=history[-1]["relative_correction"],
                                  nonlinear_residual_l2=float(np.linalg.norm(residual)), newton_seconds=seconds))
                counts.append(len(history))
                previous = state
                if physical and (target in (.5, 1.) or step == count):
                    saved_states[f"{target:g}"] = state.copy()
            result = _summary(data, initial, previous, counts, rows, steps, time.perf_counter() - started,
                              scheme=scheme, dt=dt, t_final=t_final, relaxation=relaxation, linear_solver="GMRES-ILU", **metadata)
            if not physical:
                result.update(data["errors"](operator, previous, t_final))
            else:
                theta = data["laws"].theta(previous)
                result.update(parameters=data["laws"].parameters, theta_min=float(theta.min()), theta_max=float(theta.max()), saved_states=saved_states)
            return result
    finally:
        operator.close()


def run_validation(dimension=2, scheme="implicit", nx=9, dt=.025, t_final=.5, time_model="stationary", workers=1, relaxation=1.):
    data = build_validation(dimension, scheme, nx, time_model, workers)
    return _fixed_run(data, dt, t_final, relaxation, scheme, dimension=dimension, nx=nx, h=1. / (nx - 1),
                      time_model=time_model, workers=workers, problem="polynomial_validation")


def run_physical_3d(dataset="Braudeau", nx=9, dt=.05, t_final=.1, workers=1, relaxation=1.):
    data = build_physical_3d(dataset, nx, workers)
    return _fixed_run(data, dt, t_final, relaxation, "implicit", dimension=3, dataset=dataset, nx=nx,
                      h=1. / (nx - 1), h_cm=2.75 / (nx - 1), length_cm=2.75, workers=workers, problem="wet_slotted_sphere")


def run_fourier_validation(nx=9, dt=.05, t_final=1., workers=1, relaxation=1.):
    """Use the nonlinear three-dimensional validation from the physical-test section."""
    _check_grid(nx, workers)
    p, t, boundary, interior = _mesh3d(nx)
    laws = fourier3d.ValidationLaws()
    mass = kpde3dmass(p, t, 1.)
    operator = OperateurNewtonParallele(p, t, laws, forme="psi", gravite="volume", workers=workers)
    data = dict(operator=operator, mass=mass, laws=laws, p=p, t=t, boundary=boundary, interior=interior,
                exact=fourier3d.exact, forcing=partial(fourier3d.forcing, laws=laws), errors=fourier3d.errors)
    return _fixed_run(data, dt, t_final, relaxation, "implicit", dimension=3, nx=nx, h=1. / (nx - 1), workers=workers,
                      time_model="fourier_exponential", problem="nonlinear_fourier_validation")


DRY_SAVE_TIMES = (0., .001, .002, .005, .01, .025, .05, .1, .2, .5, 1., 2., 3., 4., 5., 6., 8., 10., 15., 20., 25., 30., 40., 50., 75., 100., 125., 150., 175., 200.)


def _dry_boundary(p, boundary, time, initial_pressure=-400., ramp_hours=20.):
    points = p[boundary]
    final = np.where(np.isclose(points[:, 0], 0.), 3., 3. * (1. - points[:, 1]))
    weight = min(max(float(time) / ramp_hours, 0.), 1.) if ramp_hours > 0 else 1.
    return initial_pressure + weight * (final - initial_pressure)


def run_physical_2d(nx=21, nz=11, t_final=1., workers=1):
    """Run a short dry-infiltration preview, with the article's adaptive controller."""
    if not np.isfinite(t_final) or t_final <= 0:
        raise ValueError("t_final must be finite and positive")
    data = build_physical_2d(nx, nz, workers)
    p, boundary, free, operator, mass, laws = (data[key] for key in ("p", "boundary", "interior", "operator", "mass", "laws"))
    config = dict(initial_pressure_cm=-400., boundary_ramp_hours=20., dt_initial_hours=.01, dt_growth=1.25,
                  dt_max_hours=2., dt_reject_factor=.5, dt_min_hours=1e-6, max_consecutive_rejections=24,
                  pressure_bound_tolerance_cm=.001, newton_tolerance=NEWTON_TOLERANCE, newton_max_iterations=MAX_NEWTON_ITERATIONS)
    initial = data["initial"].copy()
    initial[boundary] = _dry_boundary(p, boundary, 0.)
    previous = initial.copy()
    thresholds = [laws.parameters[key] for key in ("SL", "AE", "LM", "MS")]
    visited = np.zeros((len(free), 5), dtype=bool)
    visited[np.arange(len(free)), np.searchsorted(thresholds, laws.theta(initial)[free], side="left")] = True
    requests = sorted(set(value for value in DRY_SAVE_TIMES if 0 < value <= t_final) | {float(t_final)})
    current, dt_next, failures, attempt = 0., config["dt_initial_hours"], 0, 0
    counts, rows, steps, attempts = [], [], [], []
    saved_states = {"0": initial.copy()}
    started = time.perf_counter()
    force = np.zeros(len(p))
    try:
        with threadpool_limits(limits=1):
            while current < t_final - 1e-11:
                next_request = next(value for value in requests if value > current + 1e-11)
                dt = min(dt_next, config["dt_max_hours"], next_request - current, t_final - current)
                if dt < config["dt_min_hours"] - 1e-12:
                    raise RuntimeError("Adaptive time step below its configured minimum")
                target = min(t_final, current + dt)
                attempt += 1
                history, state, error = [], None, None
                seconds = 0.
                try:
                    trace = _dry_boundary(p, boundary, target)
                    state, history, seconds = newton_step(operator, previous, dt, force, boundary, free, trace, mass, linear_method="direct")
                    if np.max(abs(state[boundary] - trace)) > 1e-10:
                        raise RuntimeError("Dirichlet trace changed")
                    head = state + p[:, 1]
                    tolerance = config["pressure_bound_tolerance_cm"]
                    if head.min() < -400. - tolerance or head.max() > 4. + tolerance:
                        raise RuntimeError("Hydraulic head outside the allowed bounds")
                except (RuntimeError, FloatingPointError, ValueError) as exception:
                    error = str(exception)
                    if isinstance(exception, NewtonFailure):
                        history = exception.rows
                accepted = error is None
                rows.extend(dict(step=len(counts) + 1, time=target, dt=dt, attempt=attempt, accepted=accepted, **row) for row in history)
                attempts.append(dict(attempt=attempt, accepted=accepted, start_time=current, target_time=target, dt=dt,
                                     iterations=len(history), error=error))
                if not accepted:
                    failures += 1
                    dt_next = dt * config["dt_reject_factor"]
                    if failures >= config["max_consecutive_rejections"]:
                        raise RuntimeError("Too many consecutive time-step rejections: " + error)
                    continue
                failures = 0
                theta = laws.theta(state)
                branches = np.searchsorted(thresholds, theta, side="left")
                visited[np.arange(len(free)), branches[free]] = True
                residual = operator.residu(state, previous, dt, force)[free]
                steps.append(dict(step=len(counts) + 1, time=target, dt=dt, newton_iterations=len(history),
                                  final_relative_correction=history[-1]["relative_correction"],
                                  nonlinear_residual_l2=float(np.linalg.norm(residual)), newton_seconds=seconds,
                                  pressure_min=float(state.min()), pressure_max=float(state.max()),
                                  theta_min=float(theta.min()), theta_max=float(theta.max()),
                                  hydraulic_head_min=float(head.min()), hydraulic_head_max=float(head.max()),
                                  free_branch_nodes=np.bincount(branches[free], minlength=5).tolist()))
                counts.append(len(history))
                previous, current = state, target
                if any(abs(target - value) < 1e-10 for value in (10., 20., 50., 100., 200., t_final)):
                    saved_states[f"{target:g}"] = state.copy()
                dt_next = min(config["dt_max_hours"], dt * config["dt_growth"])
            final_theta = laws.theta(previous)
            return _summary(data, initial, previous, counts, rows, steps, time.perf_counter() - started,
                            dimension=2, scheme="implicit", dataset="Braudeau", problem="dry_infiltration", nx=nx, nz=nz,
                            h_cm=[2. / (nx - 1), 1. / (nz - 1)], t_final=t_final, workers=workers, relaxation=1.,
                            linear_solver="sparse direct", parameters=laws.parameters, controller=config,
                            attempts=attempts, rejected_attempts_count=sum(not row["accepted"] for row in attempts),
                            theta_min=float(final_theta.min()), theta_max=float(final_theta.max()),
                            free_visited_branch_nodes=visited.sum(axis=0).tolist(),
                            free_nodes_visited_all_five_branches=int(visited.all(axis=1).sum()),
                            top_bottom_boundary="zero total water flux", pressure_clipping=False, saved_states=saved_states)
    finally:
        operator.close()
