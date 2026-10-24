"""Fully implicit native triangular P1 operator and exact discrete Jacobian.

All coefficients and all spatial gradients use the current Newton iterate.
The strong PDE is psi_t-g1 div_x(k psi_x)-g2 div_z(ell psi_z)
-g2 ell'(psi) psi_z = f.  Assembly uses identical nodal coefficient means,
lumped accumulation and ordered element reduction to the 3D implicit solver.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from .triangles import OperateurSemiImplicite2D, _mean, mesh, mass_matrix


class OperateurNewtonParallele(OperateurSemiImplicite2D):
    """Reuse geometry/thread lifecycle only; replace every nonlinear tensor."""
    def configuration(self):
        return dict(operator='exact_discrete_2D_implicit_newton_threaded_triangles',
                    dimension=2, coordinates='x,z', element='native triangular P1',
                    workers=self.workers, block_elements=self.block_elements,
                    blocks=len(self._blocks), forme=self.forme, gravite=self.gravite,
                    current='all coefficients and gradient(u)', frozen='none',
                    accumulation='lumped nodal m_i*(u_i-up_i)',
                    source_time='current time t_n',
                    reduction='one global reduction in fixed triangle order',
                    linear_solver='configured by the example runner',
                    native_threads='one; enforced by the example runner')

    def residu(self, u, up, dt, f):
        c = self.lois.coefficients(u)
        f = np.broadcast_to(np.asarray(f, dtype=float), (self.n,))

        def tensor(block):
            cells, vol, G = self.t[block], self.vol[block], self.G[block]
            grad = np.einsum('ej,ejk->ek', u[cells], G)
            gx, gz = grad[:, 0], grad[:, 1]
            source = (_mean(c['kdg1'], cells)*gx**2
                      +_mean(c['ldg2'], cells)*gz**2
                      -_mean(c['g2lp'], cells)*gz-_mean(f, cells))
            coefficient = np.column_stack([_mean(c['kg1'], cells),
                                           _mean(c['lg2'], cells)])
            flux = np.einsum('ek,ejk->ej', grad*coefficient, G)
            return dt*(vol[:, None]*flux+(vol/3)[:, None]*source[:, None])

        local = np.concatenate(self._ordered_tensors(tensor), axis=0)
        return (np.bincount(self.t.ravel(), weights=local.ravel(), minlength=self.n)
                +self.m*(u-up))

    def jacobien(self, u, up, dt):
        c, d = self.lois.coefficients(u), self.lois.derivees(u)

        def tensor(block):
            cells, vol, G = self.t[block], self.vol[block], self.G[block]
            grad = np.einsum('ej,ejk->ek', u[cells], G)
            gx, gz = grad[:, 0], grad[:, 1]
            ap = _mean(c['kdg1'], cells)
            az = _mean(c['ldg2'], cells)
            ag = _mean(c['g2lp'], cells)
            dsource = (d['kdg1'][cells]*gx[:, None]**2
                       +d['ldg2'][cells]*gz[:, None]**2
                       -d['g2lp'][cells]*gz[:, None])/3
            dsource += (2*ap[:, None]*gx[:, None]*G[:, :, 0]
                        +(2*az*gz-ag)[:, None]*G[:, :, 1])
            K = (_mean(c['kg1'], cells)[:, None, None]*G[:, :, None, 0]*G[:, None, :, 0]
                 +_mean(c['lg2'], cells)[:, None, None]*G[:, :, None, 1]*G[:, None, :, 1])
            K += ((gx[:, None]*G[:, :, 0])[:, :, None]*d['kg1'][cells][:, None, :]/3
                  +(gz[:, None]*G[:, :, 1])[:, :, None]*d['lg2'][cells][:, None, :]/3)
            return dt*(vol[:, None, None]*K+(vol/3)[:, None, None]*dsource[:, None, :])

        local = np.concatenate(self._ordered_tensors(tensor), axis=0)
        rows = np.broadcast_to(self.t[:, :, None], local.shape).ravel()
        cols = np.broadcast_to(self.t[:, None, :], local.shape).ravel()
        return (coo_matrix((local.ravel(), (rows, cols)), shape=(self.n, self.n)).tocsr()
                +diags(self.m))


OperateurImplicite2D = OperateurNewtonParallele
