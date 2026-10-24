"""Polynomial reference, manufactured source and P1 error quadrature in 2D."""
import numpy as np
from .smooth import Implicit3DLaws as ConstitutiveLaws


def exact(points,time,time_model='stationary',reference='poly'):
    points=np.asarray(points,dtype=float);x,z=points[...,0],points[...,1]
    if reference!='poly':raise ValueError('Only the user-specified polynomial reference is used')
    a=x*(1-x)**2;da=1-4*x+3*x*x;dda=-4+6*x
    b=2*z-6*z*z+4*z**3;db=2-12*z+12*z*z;ddb=-12+24*z
    if time_model not in ('stationary','cubic'):raise ValueError('Unknown time model')
    factor=np.exp(-time**3) if time_model=='cubic' else 1.
    u=factor*a*b
    gradient=factor*np.stack([da*b,a*db],axis=-1)
    diagonal_hessian=factor*np.stack([dda*b,a*ddb],axis=-1)
    return u,gradient,diagonal_hessian,(-3*time*time*u if time_model=='cubic' else np.zeros_like(u))


def forcing(points,time,laws=None,time_model='stationary',reference='poly'):
    laws=ConstitutiveLaws() if laws is None else laws
    u,g,h,ut=exact(points,time,time_model,reference);q=laws.fields(u)
    k,g1,g2,ell=q['k'],q['g1'],q['g2'],q['ell']
    return (ut-g1.v*(k.v*h[...,0]+k.d*g[...,0]**2)
            -g2.v*(ell.v*h[...,1]+ell.d*(g[...,1]**2+g[...,1])))


def duffy_rule(order=6):
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    a,b=np.meshgrid(x,x,indexing='ij');wa,wb=np.meshgrid(w,w,indexing='ij')
    bary=np.column_stack([((1-a)*(1-b)).ravel(),a.ravel(),((1-a)*b).ravel()])
    weights=(wa*wb*(1-a)).ravel()
    assert np.all(weights>0) and abs(weights.sum()-.5)<1e-14
    return bary,weights


def errors(op,u,time,order=6,block_size=2048,time_model='stationary',reference='poly'):
    bary,w=duffy_rule(order);l1=l2=grad2=0.;linf=0.
    for start in range(0,len(op.t),block_size):
        cells=op.t[start:start+block_size]
        points=np.einsum('qj,ejk->eqk',bary,op.p[cells])
        ue,ge,_,_=exact(points,time,time_model,reference)
        uh=np.einsum('qj,ej->eq',bary,u[cells])
        gh=np.einsum('ej,ejk->ek',u[cells],op.G[start:start+block_size])
        error=uh-ue;gradient_error=gh[:,None,:]-ge
        scale=2*op.vol[start:start+block_size]
        l1+=float(scale@(abs(error)@w));l2+=float(scale@((error**2)@w))
        grad2+=float(scale@(np.sum(gradient_error**2,axis=2)@w))
        linf=max(linf,float(np.max(abs(error))))
    return dict(error_L1=l1,error_L2=float(np.sqrt(l2)),error_H1=float(np.sqrt(l2+grad2)),
        error_H1_seminorm=float(np.sqrt(grad2)),error_Linf_quadrature_sample=linf,
        error_Linf_nodes=float(np.max(abs(u-exact(op.p,time,time_model,reference)[0]))),
        quadrature='positive tensor Gauss-Legendre/Duffy on native triangles; determinant 2*area',
        quadrature_order=order,points_per_triangle=order**2,
        Linf_scope='separate nodal and quadrature sample maxima; not certified analytic-error supremum')
