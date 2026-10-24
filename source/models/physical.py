"""Shrinkage and hydraulic laws, with the two parameter sets used in the article."""
import numpy as np

COMMON = dict(ALPHA=0.0262, N=2.088, KS=0.0095, TS=1.26, SL=0.191, ESL=0.32, AE=0.37)
DATASETS = {
    "Braudeau": dict(TR=0.064, EAE=0.4053, LM=1.176, ELM=1.32414, MS=1.224, EMS=1.36, KR=1.14, K0=0.465),
    "Garnier": dict(TR=0.0, EAE=0.420, LM=1.150, ELM=1.221, MS=1.195, EMS=1.276, KR=1.039, K0=1.134),
}

def parameters(dataset="Braudeau"):
    if dataset not in DATASETS:
        raise ValueError("dataset must be Braudeau or Garnier")
    result = dict(COMMON, **DATASETS[dataset])
    result["M"] = 1 - 1 / result["N"]
    return result

def laws(u, parameter_values):
    ALPHA, N, KS, M, TR, TS, SL, ESL, AE, EAE, LM, ELM, MS, EMS, KR, K0 = (parameter_values[key] for key in ("ALPHA", "N", "KS", "M", "TR", "TS", "SL", "ESL", "AE", "EAE", "LM", "ELM", "MS", "EMS", "KR", "K0"))
    u = np.asarray(u, dtype=float)
    a = np.abs(u)
    x = (ALPHA * a)**N
    se = (1 + x)**(-M)
    ds = -M*N*ALPHA**N*a**(N-1)*np.sign(u)*(1+x)**(-M-1)
    dds = (-M*N*ALPHA**N*(N-1)*a**(N-2)*(1+x)**(-M-1)
           + M*N*ALPHA**N*(M+1)*N*ALPHA**N*a**(2*N-2)*(1+x)**(-M-2))
    theta, dt, ddt = TR+(TS-TR)*se, (TS-TR)*ds, (TS-TR)*dds
    e, de, dde = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)
    for mask, kind in [(theta <= SL, 0), ((theta > SL)&(theta <= AE), 1),
                       ((theta > AE)&(theta <= LM), 2),
                       ((theta > LM)&(theta <= MS), 3), (theta > MS, 4)]:
        q, dq, ddq = theta[mask], dt[mask], ddt[mask]
        if kind == 0:
            e[mask] = ESL
        elif kind == 1:
            v = (q-SL)/(AE-SL)
            dv, ddv = dq/(AE-SL), ddq/(AE-SL)
            factor = KR*(AE-SL)/(np.e-1)
            e[mask] = ESL+factor*(np.exp(v)-1-v)
            de[mask] = factor*dv*(np.exp(v)-1)
            dde[mask] = factor*(ddv*np.exp(v)+dv**2*np.exp(v)-ddv)
        elif kind == 2:
            e[mask], de[mask], dde[mask] = EAE+KR*(q-AE), KR*dq, KR*ddq
        elif kind == 3:
            v, term = (q-MS)/(LM-MS), (q-LM)/(LM-MS)
            dv, ddv = dq/(LM-MS), ddq/(LM-MS)
            factor = (LM-MS)/(np.e-1)
            e[mask] = ELM+factor*((KR-K0)*(np.exp(v)-np.e)-term*(KR-K0*np.e))
            de[mask] = factor*((KR-K0)*dv*np.exp(v)-dv*(KR-K0*np.e))
            dde[mask] = factor*((KR-K0)*(ddv+dv**2)*np.exp(v)-ddv*(KR-K0*np.e))
        else:
            e[mask], de[mask], dde[mask] = EMS+K0*(q-MS), K0*dq, K0*ddq
    g1, dg1, d2g1 = 1+e, de, dde
    gz = (1+ESL)/g1
    dgz = -(1+ESL)*de/g1**2
    d2gz = -(1+ESL)*(g1*dde-2*de**2)/g1**3
    g2 = g1*gz
    dg2 = dg1*gz+g1*dgz
    d2g2 = d2g1*gz+2*dg1*dgz+g1*d2gz
    
    neg = u < 0
    s = np.clip(np.where(neg, se, 1), 1e-8, 1-1e-8)
    sp, spp = np.where(neg, ds, 0), np.where(neg, dds, 0)
    om = 1-s**(1/M)
    ss, bb = 1-om**M, om**(M-1)
    ka = .5*s**(-.5)*ss**2+2*ss*bb*s**(1/M-.5)
    kat = (-.25*s**(-1.5)*ss**2+(2/M)*ss*bb*s**(1/M-1.5)
           +2*bb**2*s**(2/M-1.5)-2*((M-1)/M)*ss*om**(M-2)*s**(2/M-1.5))
    kh = KS*s**.5*ss**2
    kh[u >= 0] = KS
    khp, khpp = KS*sp*ka, KS*(spp*ka+sp**2*kat)
    ell = kh*gz**2
    ellp = khp*gz**2+2*kh*dgz*gz
    ellpp = khpp*gz**2+4*khp*dgz*gz+2*kh*d2gz*gz+2*kh*dgz**2
    c = np.where(u <= 0, (TS-TR)*M*N*ALPHA*(ALPHA*a)**(N-1)/(1+x)**(M+1), 0)
    c = np.maximum(c,1e-8)
    cp = np.where(neg, -ALPHA**2*M*N*(TS-TR)*(ALPHA*a)**(N-2)
                  *((N-1)-(N*M+1)*x)/(1+x)**(M+2), 0)
    return dict(g1=g1,dg1=dg1,d2g1=d2g1,gz=gz,dgz=dgz,d2gz=d2gz,
                g2=g2,dg2=dg2,d2g2=d2g2,kh=kh,khp=khp,khpp=khpp,
                ell=ell,ellp=ellp,ellpp=ellpp,c=c,cp=cp)

def initial(p):
    x,y,z = p.T
    r2 = (x-.5)**2+(y-.5)**2+(z-.5)**2
    mask = ((np.abs(y-.5) > .09)|(z > .5))&(r2 <= .1)
    return -5+4*(1-np.exp(r2-.1))/(1-np.exp(-.1))*mask

class PhysicalLaws:
    """Physical coefficients with constant saturated values for positive pressure."""

    def __init__(self, dataset="Braudeau"):
        self.parameters = parameters(dataset)
        self.dataset = dataset

    _DERIVEES = ("dg1", "d2g1", "dgz", "d2gz", "dg2", "d2g2", "khp", "khpp",
                 "ellp", "ellpp", "cp")

    def lois(self, u):
        u = np.asarray(u, dtype=float)
        q = laws(np.minimum(u, 0.0), self.parameters)
        sat = u > 0
        if np.any(sat):
            for k in self._DERIVEES:
                q[k] = np.where(sat, 0.0, q[k])
        return q

    def theta(self, u):
        u = np.asarray(u, dtype=float)
        se = np.where(u < 0, (1 + (self.parameters["ALPHA"] * np.abs(u))**self.parameters["N"])**(-self.parameters["M"]), 1.0)
        return self.parameters["TR"] + (self.parameters["TS"] - self.parameters["TR"]) * se

    def dtheta(self, u):
        return self.lois(u)["c"]

    def d2theta(self, u):
        return self.lois(u)["cp"]

    def coefficients(self, u):
        q = self.lois(u)
        return dict(kg1=q["kh"] * q["g1"], lg2=q["ell"] * q["g2"],
                    kdg1=q["kh"] * q["dg1"], ldg2=q["ell"] * q["dg2"],
                    g2lp=q["g2"] * q["ellp"])

    def derivees(self, u):
        """Differentiate the coefficient products with respect to pressure."""
        q = self.lois(u)
        return dict(kg1=q["khp"] * q["g1"] + q["kh"] * q["dg1"],
                    lg2=q["ellp"] * q["g2"] + q["ell"] * q["dg2"],
                    kdg1=q["khp"] * q["dg1"] + q["kh"] * q["d2g1"],
                    ldg2=q["ellp"] * q["dg2"] + q["ell"] * q["d2g2"],
                    g2lp=q["dg2"] * q["ellp"] + q["g2"] * q["ellpp"])

