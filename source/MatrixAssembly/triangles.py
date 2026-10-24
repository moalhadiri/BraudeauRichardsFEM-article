"""Native triangular P1 FEM and exact Newton Jacobian of the 2D SSI2 model.

Coordinates are (x,z).  g1(up), g2(up), their derivatives and grad(up) stay
fixed during Newton; k(u), ell(u) and grad(u) use the current iterate.  Ordered
element blocks produce independent tensors and one global reduction.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.sparse import coo_matrix, diags


def mesh(nx):
    """Uniform unit square, two genuine triangles per Cartesian cell."""
    if isinstance(nx, bool) or int(nx) != nx or nx < 2:
        raise ValueError('nx must be an integer at least two')
    nx = int(nx)
    coordinate = np.linspace(0., 1., nx)
    x, z = np.meshgrid(coordinate, coordinate, indexing='ij')
    p = np.column_stack([x.ravel(), z.ravel()])
    left, lower = np.meshgrid(np.arange(nx-1), np.arange(nx-1), indexing='ij')
    a = (left*nx+lower).ravel()
    b, c, d = a+nx, a+nx+1, a+1
    t = np.stack([np.column_stack([a, b, c]), np.column_stack([a, c, d])],
                 axis=1).reshape(-1, 3)
    mask = np.any((p == 0.) | (p == 1.), axis=1)
    boundary = np.flatnonzero(mask)
    interior = np.flatnonzero(~mask)
    return p, t, boundary, interior


def _geometry(p, t):
    p = np.asarray(p, dtype=float)
    t = np.asarray(t)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('p must contain finite (x,z) coordinates')
    if t.ndim != 2 or t.shape[1] != 3 or len(t) == 0:
        raise ValueError('t must contain nonempty triangles')
    if not np.issubdtype(t.dtype, np.integer) or t.min() < 0 or t.max() >= len(p):
        raise ValueError('Triangle indices must be valid integers')
    vertices = p[t]
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    determinant = ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])
                   -(c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1]))
    if not np.isfinite(determinant).all() or np.any(determinant == 0):
        raise ValueError('Degenerate or nonfinite triangle geometry')
    G = np.stack([np.column_stack([b[:, 1]-c[:, 1], c[:, 0]-b[:, 0]]),
                  np.column_stack([c[:, 1]-a[:, 1], a[:, 0]-c[:, 0]]),
                  np.column_stack([a[:, 1]-b[:, 1], b[:, 0]-a[:, 0]])], axis=1)
    G /= determinant[:, None, None]
    return p, t, abs(determinant)/2, G


def mass_matrix(p, t):
    """Consistent P1 mass, diagonal area/6 and off-diagonal area/12."""
    p, t, vol, _ = _geometry(p, t)
    local = vol[:, None, None]*(np.ones((3, 3))+np.eye(3))[None, :, :]/12
    rows = np.broadcast_to(t[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(t[:, None, :], local.shape).ravel()
    return coo_matrix((local.ravel(), (rows, cols)), shape=(len(p), len(p))).tocsr()


def _mean(values, cells):
    return (values[cells[:, 0]]+values[cells[:, 1]]+values[cells[:, 2]])/3.


class OperateurSemiImplicite2D:
    """Threaded element assembly of the 2D psi_t / volume-gravity SSI2 model.

    The previous state is immutable during a Newton solve.  Passing a new
    accepted array refreshes the frozen coefficients for the next time step.
    The example runner limits native numerical libraries to one thread.
    """
    def __init__(self, p, t, lois, forme='psi', gravite='volume', workers=2):
        if forme != 'psi' or gravite != 'volume':
            raise ValueError('This operator implements psi_t with volume gravity')
        if isinstance(workers, bool) or int(workers) != workers or workers < 1:
            raise ValueError('workers must be a positive integer')
        self.p, self.t, self.vol, self.G = _geometry(p, t)
        self.n, self.lois = len(self.p), lois
        self.forme, self.gravite, self.workers = forme, gravite, int(workers)
        self.m = np.bincount(self.t.ravel(), weights=np.repeat(self.vol/3, 3),
                             minlength=self.n)
        self.block_elements = max(1, min(8192, (len(self.t)+31)//32))
        self._blocks = tuple(slice(start, min(start+self.block_elements, len(self.t)))
                             for start in range(0, len(self.t), self.block_elements))
        self._executor = ThreadPoolExecutor(max_workers=self.workers,
                                            thread_name_prefix='newton-triangles')
        self._closed = False
        self._previous_object = None

    def close(self):
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('This operator has been closed')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def configuration(self):
        return dict(operator='exact_discrete_2D_semi_implicit_newton_threaded_triangles',
                    dimension=2, coordinates='x,z', element='native triangular P1',
                    workers=self.workers, block_elements=self.block_elements,
                    blocks=len(self._blocks), forme=self.forme, gravite=self.gravite,
                    frozen='g1(up),g2(up),g1prime(up),g2prime(up),gradient(up)',
                    current='k(u),ell(u),gradient(u)',
                    accumulation='lumped nodal m_i*(u_i-up_i)',
                    source_time='previous time t_(n-1)',
                    reduction='one global reduction in fixed triangle order',
                    linear_solver='configured by the example runner',
                    native_threads='one; enforced by the example runner')

    def _ordered_tensors(self, function):
        if self._closed:
            raise RuntimeError('This operator has been closed')
        return list(self._executor.map(function, self._blocks))

    def _champs(self, values):
        return np.einsum('ej,ejk->ek', values[self.t], self.G)

    def _freeze(self, up):
        if self._previous_object is not up:
            fields = self.lois.fields(up)
            self._frozen = (fields['g1'].v, fields['g2'].v,
                            fields['g1'].d, fields['g2'].d, self._champs(up))
            self._previous_object = up
        return self._frozen

    def residu(self, u, up, dt, f):
        old1, old2, dold1, dold2, oldgrad = self._freeze(up)
        current = self.lois.fields(u)
        k, ell = current['k'], current['ell']
        hz, vz = k.v*old1, ell.v*old2
        ap, az, ag = k.v*dold1, ell.v*dold2, old2*ell.d
        f = np.broadcast_to(np.asarray(f, dtype=float), (self.n,))

        def tensor(block):
            cells, vol, G = self.t[block], self.vol[block], self.G[block]
            grad = np.einsum('ej,ejk->ek', u[cells], G)
            previous = oldgrad[block]
            source = (_mean(ap, cells)*previous[:, 0]*grad[:, 0]
                      +_mean(az, cells)*previous[:, 1]*grad[:, 1]
                      -_mean(ag, cells)*grad[:, 1]-_mean(f, cells))
            coefficient = np.column_stack([_mean(hz, cells), _mean(vz, cells)])
            flux = np.einsum('ek,ejk->ej', grad*coefficient, G)
            return dt*(vol[:, None]*flux+(vol/3)[:, None]*source[:, None])

        local = np.concatenate(self._ordered_tensors(tensor), axis=0)
        return (np.bincount(self.t.ravel(), weights=local.ravel(), minlength=self.n)
                +self.m*(u-up))

    def jacobien(self, u, up, dt):
        old1, old2, dold1, dold2, oldgrad = self._freeze(up)
        current = self.lois.fields(u)
        k, ell = current['k'], current['ell']
        hz, vz = k.v*old1, ell.v*old2
        dhz, dvz = k.d*old1, ell.d*old2
        ap, az, ag = k.v*dold1, ell.v*dold2, old2*ell.d
        dap, daz, dag = k.d*dold1, ell.d*dold2, old2*ell.dd

        def tensor(block):
            cells, vol, G = self.t[block], self.vol[block], self.G[block]
            grad = np.einsum('ej,ejk->ek', u[cells], G)
            previous = oldgrad[block]
            dsource = (dap[cells]*(previous[:, 0]*grad[:, 0])[:, None]
                       +daz[cells]*(previous[:, 1]*grad[:, 1])[:, None]
                       -dag[cells]*grad[:, 1, None])/3
            dsource += (_mean(ap, cells)[:, None]*previous[:, 0, None]*G[:, :, 0]
                        +(_mean(az, cells)*previous[:, 1]-_mean(ag, cells))[:, None]*G[:, :, 1])
            K = (_mean(hz, cells)[:, None, None]*G[:, :, None, 0]*G[:, None, :, 0]
                 +_mean(vz, cells)[:, None, None]*G[:, :, None, 1]*G[:, None, :, 1])
            flux_x, flux_z = grad[:, 0, None]*G[:, :, 0], grad[:, 1, None]*G[:, :, 1]
            K += (flux_x[:, :, None]*dhz[cells][:, None, :]/3
                  +flux_z[:, :, None]*dvz[cells][:, None, :]/3)
            return dt*(vol[:, None, None]*K+(vol/3)[:, None, None]*dsource[:, None, :])

        local = np.concatenate(self._ordered_tensors(tensor), axis=0)
        rows = np.broadcast_to(self.t[:, :, None], local.shape).ravel()
        cols = np.broadcast_to(self.t[:, None, :], local.shape).ravel()
        return (coo_matrix((local.ravel(), (rows, cols)), shape=(self.n, self.n)).tocsr()
                +diags(self.m))
