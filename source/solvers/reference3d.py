"""Sequential residual and exact Jacobian used to check the threaded assembly."""
import numpy as np
from scipy.sparse import coo_matrix, diags
from ..basisfunctions.tetrahedra import kpde3dgphi
def _mean(v, t):
    return (v[t[:, 0]] + v[t[:, 1]] + v[t[:, 2]] + v[t[:, 3]]) / 4.0

class OperateurNewton:
    def __init__(self, p, t, lois, bord=None, forme="mixte", gravite="volume"):
        """Set up geometry, mass weights and an optional nonlinear boundary term."""
        self.p, self.t, self.lois = p, t, lois
        self.bord, self.forme, self.gravite = bord, forme, gravite

        vol, g1, g2, g3, g4, _ = kpde3dgphi(p, t)
        self.vol = vol
        self.m = np.bincount(t.ravel(), weights=np.repeat(vol / 4.0, 4),
                             minlength=len(p))           
        self.G = np.stack([g1, g2, g3, g4], axis=1)        
        self.n = len(p)

    def _champs(self, u):
        G = self.G
        grad = np.einsum("ej,ejk->ek", u[self.t], G)       
        return grad

    def residu(self, u, up, tau, f):
        t, vol, G, lo = self.t, self.vol, self.G, self.lois
        c = lo.coefficients(u)
        grad = self._champs(u)
        gp2 = grad[:, 0]**2 + grad[:, 1]**2
        gz = grad[:, 2]
        hz, vz = _mean(c["kg1"], t), _mean(c["lg2"], t)
        if self.gravite == "flux":
            S = _mean(c["kdg1"], t) * gp2 + _mean(c["ldg2"], t) * gz * (gz + 1.0)
            gzf = gz + 1.0
        else:
            S = _mean(c["kdg1"], t) * gp2 + _mean(c["ldg2"], t) * gz**2 \
                - _mean(c["g2lp"], t) * gz
            gzf = gz
        src = tau * (S - _mean(f, t))
        gflux = np.c_[grad[:, 0], grad[:, 1], gzf]
        flux = np.einsum("ek,ejk->ej", gflux * np.c_[hz, hz, vz], G)  
        loc = (vol / 4.0)[:, None] * src[:, None] + tau * vol[:, None] * flux
        R = np.bincount(t.ravel(), weights=loc.ravel(), minlength=self.n)
        if self.bord is not None:
            R = R + tau * self.bord.valeur(u)
        if self.forme == "charge":
            return R + self.m * lo.dtheta(u) * (u - up)
        if self.forme == "psi":
            
            return R + self.m * (u - up)
        return R + self.m * (lo.theta(u) - lo.theta(up))

    def jacobien(self, u, up, tau):
        t, vol, G, lo = self.t, self.vol, self.G, self.lois
        c = lo.coefficients(u)
        d = lo.derivees(u)                    
        grad = self._champs(u)
        gp2 = grad[:, 0]**2 + grad[:, 1]**2
        gz = grad[:, 2]
        hz, vz = _mean(c["kg1"], t), _mean(c["lg2"], t)
        ap, az, ag = _mean(c["kdg1"], t), _mean(c["ldg2"], t), _mean(c["g2lp"], t)
        
        dhz = d["kg1"][t] / 4.0
        dvz = d["lg2"][t] / 4.0
        if self.gravite == "flux":
            dS_coef = (d["kdg1"][t] * gp2[:, None]
                       + d["ldg2"][t] * (gz * (gz + 1.0))[:, None]) / 4.0
            cz = az * (2 * gz + 1.0)
            gzf = gz + 1.0
        else:
            dS_coef = (d["kdg1"][t] * gp2[:, None] + d["ldg2"][t] * gz[:, None]**2
                       - d["g2lp"][t] * gz[:, None]) / 4.0            
            cz = 2 * az * gz - ag
            gzf = gz
        
        dS_grad = (2 * ap[:, None] * (grad[:, None, 0] * G[:, :, 0] + grad[:, None, 1] * G[:, :, 1])
                   + cz[:, None] * G[:, :, 2])                        
        
        src_j = tau * (dS_coef + dS_grad)
        Jloc = (vol / 4.0)[:, None, None] * np.broadcast_to(src_j[:, None, :], (len(t), 4, 4))

        Gp = G[:, :, :2]
        K = (hz[:, None, None] * np.einsum("eik,ejk->eij", Gp, Gp)
             + vz[:, None, None] * G[:, :, None, 2] * G[:, None, :, 2])
        fu_p = np.einsum("ek,eik->ei", grad[:, :2], Gp)               
        fu_z = gzf[:, None] * G[:, :, 2]                              
        K += fu_p[:, :, None] * dhz[:, None, :] + fu_z[:, :, None] * dvz[:, None, :]
        Jloc = Jloc + tau * vol[:, None, None] * K
        rows = np.broadcast_to(t[:, :, None], Jloc.shape).ravel()
        cols = np.broadcast_to(t[:, None, :], Jloc.shape).ravel()
        J = coo_matrix((Jloc.ravel(), (rows, cols)), shape=(self.n, self.n)).tocsr()
        if self.bord is not None:
            J = J + diags(tau * self.bord.derivee(u))
        if self.forme == "charge":
            return J + diags(self.m * (lo.d2theta(u) * (u - up) + lo.dtheta(u)))
        if self.forme == "psi":
            return J + diags(self.m)
        return J + diags(self.m * lo.dtheta(u))

