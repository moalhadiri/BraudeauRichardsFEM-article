"""Fully implicit physical 2D psi equation with zero total vertical flux.

The inherited volume terms are unchanged from the validated implicit 2D
operator.  On the horizontal edges the condition is ell(psi)*(psi_z+1)=0,
so its weak-form boundary contribution is +dt*integral(g2*ell*n_z*v).
This contribution uses endpoint lumping and its exact discrete derivative.
Coordinates are physical centimetres, ordered (x,z).
"""
import numpy as np
from scipy.sparse import diags

from .implicit2d import OperateurNewtonParallele


def rectangle_mesh(nx, nz, Lx=2.0, Lz=1.0):
    """Return native P1 points/triangles, lateral Dirichlet and free nodes.

    Only x=0 and x=Lx are Dirichlet; nodes on z=0 or z=Lz remain free,
    except for their intersections with the prescribed lateral edges.
    """
    for name, count in (("nx", nx), ("nz", nz)):
        if isinstance(count, bool) or int(count) != count or count < 2:
            raise ValueError(f"{name} must be an integer at least two")
    if not np.isfinite([Lx, Lz]).all() or Lx <= 0 or Lz <= 0:
        raise ValueError("Lx and Lz must be finite positive lengths")
    nx, nz = int(nx), int(nz)
    xx, zz = np.meshgrid(np.linspace(0.0, float(Lx), nx),
                         np.linspace(0.0, float(Lz), nz), indexing="ij")
    p = np.column_stack((xx.ravel(), zz.ravel()))
    ix, iz = np.meshgrid(np.arange(nx-1), np.arange(nz-1), indexing="ij")
    a = (ix*nz+iz).ravel()
    b, c, d = a+nz, a+nz+1, a+1
    triangles = np.stack((np.column_stack((a, b, c)),
                          np.column_stack((a, c, d))), axis=1).reshape(-1, 3)
    mask = (p[:, 0] == 0.0) | (p[:, 0] == Lx)
    return p, triangles, np.flatnonzero(mask), np.flatnonzero(~mask)


def horizontal_flux_weights(p, triangles):
    """Endpoint weights for integral(n_z*v) on the horizontal boundary.

    Extract edges from the actual triangle topology, rather than assuming
    the structured node numbering.  Lateral exterior edges are Dirichlet.
    """
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)],
                            triangles[:, (2, 0)]), axis=0)
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold triangle edge")
    exterior = unique[counts == 1]
    lo, hi = p.min(axis=0), p.max(axis=0)
    tolerance = 64*np.finfo(float).eps*max(1.0, float(np.max(hi-lo)))
    weights = np.zeros(len(p))
    horizontal_edges = []
    normals = []
    for edge in exterior:
        endpoints = p[edge]
        if np.all(abs(endpoints[:, 1]-lo[1]) <= tolerance):
            normal = -1.0
        elif np.all(abs(endpoints[:, 1]-hi[1]) <= tolerance):
            normal = 1.0
        elif (np.all(abs(endpoints[:, 0]-lo[0]) <= tolerance)
              or np.all(abs(endpoints[:, 0]-hi[0]) <= tolerance)):
            continue
        else:
            raise ValueError("Physical 2D boundary must be an axis-aligned rectangle")
        length = float(np.linalg.norm(endpoints[1]-endpoints[0]))
        weights[edge] += normal*length/2.0
        horizontal_edges.append(edge)
        normals.append(normal)
    if not horizontal_edges:
        raise ValueError("No horizontal exterior edges found")
    return weights, np.asarray(horizontal_edges, dtype=int), np.asarray(normals)


class PhysicalImplicit2D(OperateurNewtonParallele):
    """Native implicit psi_t physical operator, parallel element assembly."""

    def __init__(self, p, t, lois, forme="psi", gravite="volume", workers=2):
        super().__init__(p, t, lois, forme=forme, gravite=gravite, workers=workers)
        try:
            self.flux_weights, self.horizontal_edges, self.edge_normals = (
                horizontal_flux_weights(self.p, self.t))
        except Exception:
            self.close()
            raise

    def configuration(self):
        description = super().configuration()
        description.update(
            operator="physical_2D_implicit_psi_exact_newton_total_flux",
            boundary_dirichlet="x=0 and x=Lx only",
            boundary_horizontal="zero total water flux: ell(psi)*(psi_z+1)=0",
            boundary_quadrature="edge endpoint lumping, exact discrete derivative",
            horizontal_boundary_edges=len(self.horizontal_edges),
            transport="none")
        return description

    def residu(self, u, up, dt, f):
        residual = super().residu(u, up, dt, f)
        return residual+dt*self.flux_weights*self.lois.coefficients(u)["lg2"]

    def jacobien(self, u, up, dt):
        jacobian = super().jacobien(u, up, dt)
        return jacobian+diags(dt*self.flux_weights*self.lois.derivees(u)["lg2"])
