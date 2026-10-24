"""Assemble the exact residual and Jacobian in ordered element blocks.

The sequential formulas are in ``source.solvers.reference3d``. Threads evaluate
separate blocks; their contributions are reduced once in mesh order. Newton
updates and linear solves are handled by ``source.solvers.newton``.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags

from ..solvers import reference3d as _reference


REFERENCE_SHA256 = "5491581abf170a0c6aa15da0727d5ac47d6f478b486b6efe77c70ed31d236126"


def verify_reference():
    """Refuse to silently parallelize a different version of the source solver."""
    path = Path(_reference.__file__).resolve()
    observed = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if observed != REFERENCE_SHA256:
        raise RuntimeError(
            "The exact Newton reference changed; re-audit the parallel assembly "
            f"before running. Expected {REFERENCE_SHA256}; got {observed}: {path}"
        )
    return observed


def _mean(v, t):
    # Keep the operation order of the reference (not np.mean or np.sum).
    return (v[t[:, 0]] + v[t[:, 1]] + v[t[:, 2]] + v[t[:, 3]]) / 4.0


class OperateurNewtonParallele(_reference.OperateurNewton):
    """Original operator with threaded residual/Jacobian element assembly.

    ``forme='psi'/'mixte'/'charge'``, ``gravite='volume'/'flux'`` and the optional
    diagonal boundary term are supported with exactly the reference formulas.
    One instance must not be evaluated concurrently by multiple time integrators.
    Call ``close()`` after the trajectory, or use the instance as a context manager.
    """

    def __init__(self, p, t, lois, bord=None, forme="mixte", gravite="volume",
                 *, workers=8, block_elements=None):
        verify_reference()
        if isinstance(workers, bool) or int(workers) != workers or workers < 1:
            raise ValueError("workers must be a positive integer")
        if block_elements is None:
            # At least 32 tasks on a small mesh; bound each large-mesh block.
            # This partition depends only on the mesh, never on worker count.
            block_elements = max(1, min(8192, (len(t) + 31) // 32))
        if (isinstance(block_elements, bool) or int(block_elements) != block_elements
                or block_elements < 1):
            raise ValueError("block_elements must be a positive integer")
        if forme not in ("psi", "mixte", "charge"):
            raise ValueError("forme must be 'psi', 'mixte' or 'charge'")
        if gravite not in ("volume", "flux"):
            raise ValueError("gravite must be 'volume' or 'flux'")
        super().__init__(p, t, lois, bord=bord, forme=forme, gravite=gravite)
        self.workers = int(workers)
        self.block_elements = int(block_elements)
        self._blocks = tuple(
            slice(start, min(start + self.block_elements, len(t)))
            for start in range(0, len(t), self.block_elements)
        )
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="newton-elements")
        self._closed = False

    def close(self):
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("This parallel Newton operator is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def configuration(self):
        return dict(
            operator="exact_discrete_newton_threaded_elements",
            reference_sha256=REFERENCE_SHA256,
            workers=self.workers,
            block_elements=self.block_elements,
            blocks=len(self._blocks),
            forme=self.forme,
            gravite=self.gravite,
            nodal_laws="one evaluation on calling thread",
            reduction="one global reduction in original element order",
            linear_solver="configured by the example runner",
        )

    def _ordered_tensors(self, function):
        if self._closed:
            raise RuntimeError("This parallel Newton operator is already closed")
        # Executor.map yields results in input order, independent of completion.
        return list(self._executor.map(function, self._blocks))

    def residu(self, u, up, tau, f):
        c = self.lois.coefficients(u)

        def element_residual(block):
            t, vol, G = self.t[block], self.vol[block], self.G[block]
            grad = np.einsum("ej,ejk->ek", u[t], G)
            gp2 = grad[:, 0]**2 + grad[:, 1]**2
            gz = grad[:, 2]
            hz, vz = _mean(c["kg1"], t), _mean(c["lg2"], t)
            if self.gravite == "flux":
                S = (_mean(c["kdg1"], t) * gp2
                     + _mean(c["ldg2"], t) * gz * (gz + 1.0))
                gzf = gz + 1.0
            else:
                S = (_mean(c["kdg1"], t) * gp2
                     + _mean(c["ldg2"], t) * gz**2
                     - _mean(c["g2lp"], t) * gz)
                gzf = gz
            src = tau * (S - _mean(f, t))
            gflux = np.c_[grad[:, 0], grad[:, 1], gzf]
            flux = np.einsum("ek,ejk->ej", gflux * np.c_[hz, hz, vz], G)
            return ((vol / 4.0)[:, None] * src[:, None]
                    + tau * vol[:, None] * flux)

        tensors = self._ordered_tensors(element_residual)
        loc = np.concatenate(tensors, axis=0) if tensors else np.empty((0, 4))
        del tensors
        R = np.bincount(self.t.ravel(), weights=loc.ravel(), minlength=self.n)
        if self.bord is not None:
            R = R + tau * self.bord.valeur(u)
        if self.forme == "charge":
            return R + self.m * self.lois.dtheta(u) * (u - up)
        if self.forme == "psi":
            return R + self.m * (u - up)
        return R + self.m * (self.lois.theta(u) - self.lois.theta(up))

    def jacobien(self, u, up, tau):
        c = self.lois.coefficients(u)
        d = self.lois.derivees(u)

        def element_jacobian(block):
            t, vol, G = self.t[block], self.vol[block], self.G[block]
            grad = np.einsum("ej,ejk->ek", u[t], G)
            gp2 = grad[:, 0]**2 + grad[:, 1]**2
            gz = grad[:, 2]
            hz, vz = _mean(c["kg1"], t), _mean(c["lg2"], t)
            ap = _mean(c["kdg1"], t)
            az = _mean(c["ldg2"], t)
            ag = _mean(c["g2lp"], t)
            dhz = d["kg1"][t] / 4.0
            dvz = d["lg2"][t] / 4.0
            if self.gravite == "flux":
                dS_coef = (d["kdg1"][t] * gp2[:, None]
                           + d["ldg2"][t] * (gz * (gz + 1.0))[:, None]) / 4.0
                cz = az * (2 * gz + 1.0)
                gzf = gz + 1.0
            else:
                dS_coef = (d["kdg1"][t] * gp2[:, None]
                           + d["ldg2"][t] * gz[:, None]**2
                           - d["g2lp"][t] * gz[:, None]) / 4.0
                cz = 2 * az * gz - ag
                gzf = gz
            dS_grad = (
                2 * ap[:, None] * (
                    grad[:, None, 0] * G[:, :, 0]
                    + grad[:, None, 1] * G[:, :, 1]
                ) + cz[:, None] * G[:, :, 2]
            )
            src_j = tau * (dS_coef + dS_grad)
            Jloc = (vol / 4.0)[:, None, None] * np.broadcast_to(
                src_j[:, None, :], (len(t), 4, 4)
            )
            Gp = G[:, :, :2]
            K = (
                hz[:, None, None] * np.einsum("eik,ejk->eij", Gp, Gp)
                + vz[:, None, None] * G[:, :, None, 2] * G[:, None, :, 2]
            )
            fu_p = np.einsum("ek,eik->ei", grad[:, :2], Gp)
            fu_z = gzf[:, None] * G[:, :, 2]
            K += (fu_p[:, :, None] * dhz[:, None, :]
                  + fu_z[:, :, None] * dvz[:, None, :])
            return Jloc + tau * vol[:, None, None] * K

        tensors = self._ordered_tensors(element_jacobian)
        Jloc = np.concatenate(tensors, axis=0) if tensors else np.empty((0, 4, 4))
        del tensors
        rows = np.broadcast_to(self.t[:, :, None], Jloc.shape).ravel()
        cols = np.broadcast_to(self.t[:, None, :], Jloc.shape).ravel()
        J = coo_matrix((Jloc.ravel(), (rows, cols)), shape=(self.n, self.n)).tocsr()
        if self.bord is not None:
            J = J + diags(tau * self.bord.derivee(u))
        if self.forme == "charge":
            return J + diags(self.m * (
                self.lois.d2theta(u) * (u - up) + self.lois.dtheta(u)
            ))
        if self.forme == "psi":
            return J + diags(self.m)
        return J + diags(self.m * self.lois.dtheta(u))
