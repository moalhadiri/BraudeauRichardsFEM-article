"""Polynomial reference, manufactured source and P1 error quadrature in 3D."""
import numpy as np
from .smooth import Implicit3DLaws
def time_factor(time, time_model='stationary'):
    if time_model == 'stationary':
        return 1.0
    if time_model == 'exponential':
        return float(np.exp(-time))
    raise ValueError('time_model must be stationary or exponential')


def exact(points, time, time_model='stationary'):
    """Return u, grad(u), diagonal Hessian and u_t on arbitrary point batches."""
    p = np.asarray(points, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    ax, az = x*(1-x)**2, z*(1-z)**2
    by = 2*y-6*y**2+4*y**3
    dx, dz, dy = 1-4*x+3*x**2, 1-4*z+3*z**2, 2-12*y+12*y**2
    dxx, dzz, dyy = -4+6*x, -4+6*z, -12+24*y
    factor = time_factor(time, time_model)
    u = factor*ax*az*by
    grad = factor*np.stack([dx*az*by, ax*az*dy, ax*dz*by], axis=-1)
    hdiag = factor*np.stack([dxx*az*by, ax*az*dyy, ax*dzz*by], axis=-1)
    ut = -u if time_model == 'exponential' else np.zeros_like(u)
    return u, grad, hdiag, ut


def forcing(points, time, laws=None, time_model='stationary'):
    """Strong manufactured source for the pressure equation with volume gravity.

    f = u_t - g1*div_xy(k*grad_xy(u))
            - g2*div_z(ell*u_z) - g2*ell'(u)*u_z.
    g1 and g2 multiply their divergences; their derivatives belong to the
    weak residual/Jacobian and must not be added to this strong source.
    """
    laws = Implicit3DLaws() if laws is None else laws
    u, grad, hdiag, ut = exact(points, time, time_model)
    q = laws.fields(u)
    k, g1, g2, ell = q['k'], q['g1'], q['g2'], q['ell']
    horizontal = g1.v*(k.v*(hdiag[..., 0]+hdiag[..., 1])
                       +k.d*(grad[..., 0]**2+grad[..., 1]**2))
    vertical = g2.v*(ell.v*hdiag[..., 2]+ell.d*(grad[..., 2]**2+grad[..., 2]))
    return ut-horizontal-vertical


def duffy_rule(order=5):
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x+1)/2, w/2
    a, b, c = np.meshgrid(x, x, x, indexing='ij')
    wa, wb, wc = np.meshgrid(w, w, w, indexing='ij')
    bary = np.column_stack([((1-a)*(1-b)*(1-c)).ravel(), a.ravel(),
                            ((1-a)*b).ravel(), ((1-a)*(1-b)*c).ravel()])
    weights = (wa*wb*wc*(1-a)**2*(1-b)).ravel()
    assert np.all(weights > 0) and abs(weights.sum()-1/6) < 1e-14
    return bary, weights


def errors(op, u, time, order=5, block_size=1024, time_model='stationary'):
    """Native P1 errors, positive quadrature and physical Jacobian 6*volume."""
    bary, w = duffy_rule(order)
    total_l1 = total_l2 = total_grad2 = 0.0
    linf_q = 0.0
    for start in range(0, len(op.t), block_size):
        cells = op.t[start:start+block_size]
        points = op.p[cells]
        qp = np.einsum('qj,ejk->eqk', bary, points)
        ue, ge, _, _ = exact(qp, time, time_model)
        uh = np.einsum('qj,ej->eq', bary, u[cells])
        gh = np.einsum('ej,ejk->ek', u[cells], op.G[start:start+block_size])
        e = uh-ue
        grad_error = gh[:, None, :]-ge
        scale = 6*op.vol[start:start+block_size]
        total_l1 += float(scale@(np.abs(e)@w))
        total_l2 += float(scale@((e*e)@w))
        total_grad2 += float(scale@(np.sum(grad_error**2, axis=2)@w))
        linf_q = max(linf_q, float(np.max(abs(e))))
    return dict(error_L1=total_l1, error_L2=float(np.sqrt(total_l2)),
                error_H1=float(np.sqrt(total_l2+total_grad2)),
                error_H1_seminorm=float(np.sqrt(total_grad2)),
                error_Linf_quadrature_sample=linf_q,
                error_Linf_nodes=float(np.max(abs(u-exact(op.p, time, time_model)[0]))),
                quadrature='positive tensor Gauss-Legendre/Duffy on native tetrahedra',
                quadrature_order=order, points_per_tetrahedron=order**3,
                Linf_scope='maximum over quadrature sample; separate nodal maximum; not certified continuous supremum')

