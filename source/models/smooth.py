"""Smooth validation coefficients and their first two state derivatives."""
import numpy as np
class Jet:
    """Value and two exact derivatives with respect to the scalar state."""
    def __init__(self, v, d=0.0, dd=0.0):
        self.v, self.d, self.dd = np.asarray(v), np.asarray(d), np.asarray(dd)

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Jet) else Jet(x)

    def __add__(self, x):
        x = self.wrap(x)
        return Jet(self.v+x.v, self.d+x.d, self.dd+x.dd)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.d, -self.dd)

    def __sub__(self, x):
        return self+-self.wrap(x)

    def __rsub__(self, x):
        return self.wrap(x)+-self

    def __mul__(self, x):
        x = self.wrap(x)
        return Jet(self.v*x.v, self.d*x.v+self.v*x.d,
                   self.dd*x.v+2*self.d*x.d+self.v*x.dd)

    __rmul__ = __mul__

    def __pow__(self, p):
        if p == 0:
            return Jet(np.ones_like(self.v), np.zeros_like(self.v), np.zeros_like(self.v))
        if p == 1:
            return Jet(self.v, self.d, self.dd)
        return Jet(self.v**p, p*self.v**(p-1)*self.d,
                   p*(p-1)*self.v**(p-2)*self.d**2+p*self.v**(p-1)*self.dd)

    def __truediv__(self, x):
        return self*self.wrap(x)**-1

    def __rtruediv__(self, x):
        return self.wrap(x)*self**-1


class Implicit3DLaws:
    """Literal smooth reference laws, with their genuine pole at u=-1.

    No capacity division, saturation clipping, or artificial denominator is
    introduced.  The 0.01 in g1 is the parameter present in the source model.
    """
    def fields(self, u):
        u = np.asarray(u, dtype=float)
        if not np.isfinite(u).all():
            raise FloatingPointError('Nonfinite state in constitutive laws')
        state = Jet(u, np.ones_like(u), np.zeros_like(u))
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            k = (1+state**2)**(-.5)
            g1 = 1+(.01+(state-.5)**2)**-1
            gz = (1+state**3)**-2
            g2 = g1*gz
            ell = k*gz
        return dict(k=k, g1=g1, gz=gz, g2=g2, ell=ell)

    def coefficients(self, u):
        q = self.fields(u)
        k, g1, g2, ell = q['k'], q['g1'], q['g2'], q['ell']
        return dict(kg1=(k*g1).v, lg2=(ell*g2).v,
                    kdg1=k.v*g1.d, ldg2=ell.v*g2.d, g2lp=g2.v*ell.d)

    # Both spellings are accepted by the finite element operators.
    coefficient = coefficients

    def derivees(self, u):
        q = self.fields(u)
        k, g1, g2, ell = q['k'], q['g1'], q['g2'], q['ell']
        return dict(kg1=(k*g1).d, lg2=(ell*g2).d,
                    kdg1=k.d*g1.d+k.v*g1.dd,
                    ldg2=ell.d*g2.d+ell.v*g2.dd,
                    g2lp=g2.d*ell.d+g2.v*ell.dd)

