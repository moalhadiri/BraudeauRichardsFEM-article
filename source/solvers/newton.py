"""Damped Newton steps and the checked linear solves used in the examples."""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu

NEWTON_TOLERANCE = 1e-8
MAX_NEWTON_ITERATIONS = 200
LINEAR_RTOL = 1e-10
TRUE_RESIDUAL_LIMIT = 1.05e-10
ILU_ATTEMPTS = (
    dict(drop_tol=1e-3, fill_factor=10, restart=50, maxiter=40),
    dict(drop_tol=1e-4, fill_factor=20, restart=100, maxiter=20),
)


class NewtonFailure(RuntimeError):
    def __init__(self, message, rows):
        super().__init__(message)
        self.rows = rows


def linear_solve(matrix, rhs, method="gmres"):
    """Check the true residual after GMRES; do not fall back to a direct solve."""
    matrix = matrix.tocsc()
    rhs = np.asarray(rhs, dtype=np.float64)
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("Nonfinite linear system")
    norm = float(np.linalg.norm(rhs))
    if norm == 0:
        return np.zeros_like(rhs), dict(linear_relative_residual=0., gmres_iterations=0, linear_attempts=0)
    if method == "direct":
        factor = splu(matrix)
        solution = factor.solve(rhs)
        error = float(np.linalg.norm(matrix @ solution - rhs) / norm)
        if not np.isfinite(solution).all():
            raise FloatingPointError("Nonfinite direct correction")
        return solution, dict(linear_relative_residual=error, gmres_iterations=0, linear_attempts=1)
    if method != "gmres":
        raise ValueError("linear method must be gmres or direct")
    total = 0
    last_error = None
    for number, options in enumerate(ILU_ATTEMPTS, start=1):
        try:
            factor = spilu(matrix, drop_tol=options["drop_tol"], fill_factor=options["fill_factor"], permc_spec="COLAMD")
        except RuntimeError as error:
            last_error = str(error)
            continue
        preconditioner = LinearOperator(matrix.shape, matvec=factor.solve, dtype=np.float64)
        inner = [0]
        def callback(_value):
            inner[0] += 1
        solution, info = gmres(matrix, rhs, M=preconditioner, rtol=LINEAR_RTOL, atol=0.0,
                               restart=min(options["restart"], matrix.shape[0]), maxiter=options["maxiter"],
                               callback=callback, callback_type="pr_norm")
        total += inner[0]
        finite = bool(np.isfinite(solution).all())
        error = float(np.linalg.norm(matrix @ solution - rhs) / norm) if finite else float("inf")
        del preconditioner, factor
        if info == 0 and finite and error <= TRUE_RESIDUAL_LIMIT:
            return solution, dict(linear_relative_residual=error, gmres_iterations=total, linear_attempts=number,
                                  preconditioner_drop_tol=options["drop_tol"], preconditioner_fill_factor=options["fill_factor"])
        last_error = f"GMRES info={info}, true relative residual={error:.3e}"
    raise RuntimeError("GMRES-ILU did not meet the true-residual tolerance: " + str(last_error))


def newton_step(operator, previous, dt, source, boundary, interior, boundary_values, mass,
                relaxation=1., tolerance=NEWTON_TOLERANCE, maximum_iterations=MAX_NEWTON_ITERATIONS,
                linear_method="gmres"):
    """Use the previous field as the starting value and test the raw correction."""
    if not 0 < relaxation <= 1 or dt <= 0 or tolerance <= 0:
        raise ValueError("Invalid Newton parameters")
    state = previous.copy()
    state[boundary] = boundary_values
    rows = []
    start = time.perf_counter()
    for iteration in range(1, maximum_iterations + 1):
        residual = operator.residu(state, previous, dt, source)
        jacobian = operator.jacobien(state, previous, dt)
        correction = np.zeros_like(state)
        correction[interior], linear = linear_solve(jacobian[interior, :][:, interior], -residual[interior], linear_method)
        state = state + relaxation * correction
        increment = np.sqrt(max(correction @ (mass @ correction), 0.0) / max(state @ (mass @ state), 1e-300))
        row = dict(iteration=iteration, relative_correction=float(increment),
                   applied_relative_correction=float(relaxation * increment),
                   residual_before_l2=float(np.linalg.norm(residual[interior])), **linear)
        rows.append(row)
        if not np.isfinite(state).all() or not np.isfinite(increment):
            raise NewtonFailure("Nonfinite Newton iterate", rows)
        if increment <= tolerance:
            return state, rows, time.perf_counter() - start
    raise NewtonFailure(f"Newton did not converge in {maximum_iterations} corrections", rows)
