"""Fourier validation with exponent three and its independently prescribed source."""
import numpy as np

EXPONENT=3.0
MODES=np.array([3.0,2.0,1.0])
BETA=1.0/36.0
AMPLITUDE=float((1-np.cos(MODES[1]*np.pi/2))*(1-np.cos(MODES[2]*np.pi/2))/np.prod(MODES))
DECAY=float(np.pi**2*np.sum(MODES**2)*BETA)


class Jet:
    """Value, first and second psi derivatives; exact elementary chain rules."""
    def __init__(self,v,d=0.0,dd=0.0):self.v,self.d,self.dd=np.asarray(v),np.asarray(d),np.asarray(dd)
    @staticmethod
    def wrap(x):return x if isinstance(x,Jet) else Jet(x)
    def __add__(self,x):
        x=self.wrap(x);return Jet(self.v+x.v,self.d+x.d,self.dd+x.dd)
    __radd__=__add__
    def __neg__(self):return Jet(-self.v,-self.d,-self.dd)
    def __sub__(self,x):return self+-self.wrap(x)
    def __rsub__(self,x):return self.wrap(x)+-self
    def __mul__(self,x):
        x=self.wrap(x)
        return Jet(self.v*x.v,self.d*x.v+self.v*x.d,self.dd*x.v+2*self.d*x.d+self.v*x.dd)
    __rmul__=__mul__
    def __pow__(self,p):
        if p==0:return Jet(np.ones_like(self.v),np.zeros_like(self.v),np.zeros_like(self.v))
        if p==1:return Jet(self.v,self.d,self.dd)
        return Jet(self.v**p,p*self.v**(p-1)*self.d,
                   p*(p-1)*self.v**(p-2)*self.d**2+p*self.v**(p-1)*self.dd)
    def __truediv__(self,x):return self*self.wrap(x)**-1
    def __rtruediv__(self,x):return self.wrap(x)*self**-1
    def exp(self):
        e=np.exp(self.v);return Jet(e,e*self.d,e*(self.d**2+self.dd))


class ValidationLaws:
    def __init__(self,exponent=EXPONENT):
        if exponent!=3:raise ValueError('Reference ell_p is exactly 3')
        self.exponent=exponent
    def fields(self,u):
        u=np.asarray(u,dtype=float)
        a=Jet(np.maximum(-u,0.0),np.where(u<0,-1.0,0.0),np.zeros_like(u))
        q=1-1/self.exponent
        se=(1+a**self.exponent)**(-q)
        # Algebraically identical Mualem law, avoiding cancellation near Se=1.
        k=se**.5*(1-a**(self.exponent-1)*se)**2
        g1=1+se.exp()-se
        gz=1/g1
        ell=k*gz**2
        return dict(se=se,k=k,g1=g1,gz=gz,ell=ell)
    def coefficients(self,u):
        q=self.fields(u);k,g,l=q['k'],q['g1'],q['ell']
        return dict(kg1=(k*g).v,lg2=l.v,kdg1=k.v*g.d,
                    ldg2=np.zeros_like(np.asarray(u,dtype=float)),g2lp=l.d)
    def derivees(self,u):
        q=self.fields(u);k,g,l=q['k'],q['g1'],q['ell']
        return dict(kg1=(k*g).d,lg2=l.d,kdg1=k.d*g.d+k.v*g.dd,
                    ldg2=np.zeros_like(np.asarray(u,dtype=float)),g2lp=l.dd)


def exact(points,time):
    """Return u, grad u, diagonal Hessian and u_t without time interpolation."""
    p=np.asarray(points,dtype=float)
    omega=np.pi*MODES
    sine=np.sin(p*omega)
    # The exact Dirichlet trace is zero, without roundoff sin(k*pi) remnants.
    sine=np.where((p==0)|(p==1),0.0,sine)
    cosine=np.cos(p*omega)
    factor=-AMPLITUDE*np.exp(-DECAY*time)
    u=factor*np.prod(sine,axis=-1)
    grad=np.stack([factor*omega[j]*cosine[...,j]*sine[...,(j+1)%3]*sine[...,(j+2)%3]
                   for j in range(3)],axis=-1)
    hdiag=-u[...,None]*omega**2
    return u,grad,hdiag,-DECAY*u


def forcing(points,time,laws=None):
    laws=ValidationLaws() if laws is None else laws
    u,g,h,ut=exact(points,time);q=laws.fields(u)
    k,g1,ell=q['k'],q['g1'],q['ell']
    # g2=g1*gz=1; gravity is the volume term -g2*ell'(u)*u_z.
    horizontal=g1.v*(k.v*(h[...,0]+h[...,1])+k.d*(g[...,0]**2+g[...,1]**2))
    vertical=ell.v*h[...,2]+ell.d*g[...,2]**2+ell.d*g[...,2]
    return ut-horizontal-vertical


def duffy_rule(order=5):
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    a,b,c=np.meshgrid(x,x,x,indexing='ij')
    wa,wb,wc=np.meshgrid(w,w,w,indexing='ij')
    bary=np.column_stack([((1-a)*(1-b)*(1-c)).ravel(),a.ravel(),
                          ((1-a)*b).ravel(),((1-a)*(1-b)*c).ravel()])
    weights=(wa*wb*wc*(1-a)**2*(1-b)).ravel()
    assert np.all(weights>0) and abs(weights.sum()-1/6)<1e-14
    return bary,weights


def errors(op,u,time,order=5,block_size=1024):
    """Positive tensor Gauss/Duffy rule, proper physical determinant scaling."""
    bary,w=duffy_rule(order)
    total_l1=total_l2=total_grad2=0.0;linf_q=0.0
    for start in range(0,len(op.t),block_size):
        cells=op.t[start:start+block_size];points=op.p[cells]
        qp=np.einsum('qj,ejk->eqk',bary,points)
        ue,ge,_,_=exact(qp,time)
        uh=np.einsum('qj,ej->eq',bary,u[cells])
        gh=np.einsum('ej,ejk->ek',u[cells],op.G[start:start+block_size])
        e=uh-ue;grad_error=gh[:,None,:]-ge
        scale=6*op.vol[start:start+block_size]
        total_l1+=float(scale@(np.abs(e)@w))
        total_l2+=float(scale@((e*e)@w))
        total_grad2+=float(scale@(np.sum(grad_error**2,axis=2)@w))
        linf_q=max(linf_q,float(np.max(abs(e))))
    return dict(error_L1=total_l1,error_L2=float(np.sqrt(total_l2)),
                error_H1=float(np.sqrt(total_l2+total_grad2)),
                error_H1_seminorm=float(np.sqrt(total_grad2)),
                error_Linf_quadrature_sample=linf_q,
                error_Linf_nodes=float(np.max(abs(u-exact(op.p,time)[0]))),
                quadrature='positive tensor Gauss-Legendre/Duffy on native tetrahedra',
                quadrature_order=order,points_per_tetrahedron=order**3,
                Linf_scope='maximum over quadrature sample; separate nodal maximum; not certified continuous supremum')
