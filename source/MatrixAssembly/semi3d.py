"""Exact Jacobian of SSI2, frozen g1/g2 and their previous-state gradients.

Reuse the checked mesh geometry, ordered element blocks and Newton loop.
Unlike the fully implicit residual, grad(g(up)) is held fixed in Newton.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from .parallel3d import OperateurNewtonParallele,_mean


class OperateurSemiImpliciteParallele(OperateurNewtonParallele):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.forme!='psi' or self.gravite!='volume' or self.bord is not None:
            raise ValueError('This SSI2 implementation is psi_t with volume gravity and Dirichlet')
        self._previous_object=None

    def configuration(self):
        result=super().configuration()
        result.update(operator='exact_discrete_semi_implicit_newton_threaded_elements',
            frozen='g1(up),g2(up),g1prime(up),g2prime(up),gradient(up)',
            current='k(u),ell(u),gradient(u)',source_time='previous time t_(n-1)')
        return result

    def _freeze(self,up):
        # up is immutable during the Newton solve; a new accepted array
        # identifies the next time step. No thread mutates these cached fields.
        if self._previous_object is not up:
            fields=self.lois.fields(up)
            self._frozen=(fields['g1'].v,fields['g2'].v,fields['g1'].d,
                          fields['g2'].d,self._champs(up))
            self._previous_object=up
        return self._frozen

    def residu(self,u,up,tau,f):
        old1,old2,dold1,dold2,oldgrad=self._freeze(up)
        current=self.lois.fields(u);k,l=current['k'],current['ell']
        hz=k.v*old1;vz=l.v*old2
        ap=k.v*dold1;az=l.v*dold2;ag=old2*l.d
        def tensor(block):
            cells,vol,G=self.t[block],self.vol[block],self.G[block]
            grad=np.einsum('ej,ejk->ek',u[cells],G);pg=oldgrad[block]
            cross=pg[:,0]*grad[:,0]+pg[:,1]*grad[:,1]
            source=(_mean(ap,cells)*cross+_mean(az,cells)*pg[:,2]*grad[:,2]
                    -_mean(ag,cells)*grad[:,2]-_mean(f,cells))
            coeff=np.column_stack([_mean(hz,cells),_mean(hz,cells),_mean(vz,cells)])
            flux=np.einsum('ek,ejk->ej',grad*coeff,G)
            return tau*(vol[:,None]*flux+(vol/4)[:,None]*source[:,None])
        local=np.concatenate(self._ordered_tensors(tensor),axis=0)
        return np.bincount(self.t.ravel(),weights=local.ravel(),minlength=self.n)+self.m*(u-up)

    def jacobien(self,u,up,tau):
        old1,old2,dold1,dold2,oldgrad=self._freeze(up)
        current=self.lois.fields(u);k,l=current['k'],current['ell']
        hz,vz=k.v*old1,l.v*old2
        dhz,dvz=k.d*old1,l.d*old2
        ap,az,ag=k.v*dold1,l.v*dold2,old2*l.d
        dap,daz,dag=k.d*dold1,l.d*dold2,old2*l.dd
        def tensor(block):
            cells,vol,G=self.t[block],self.vol[block],self.G[block]
            grad=np.einsum('ej,ejk->ek',u[cells],G);pg=oldgrad[block]
            cross=pg[:,0]*grad[:,0]+pg[:,1]*grad[:,1]
            dsource=(dap[cells]*cross[:,None]
                     +daz[cells]*(pg[:,2]*grad[:,2])[:,None]
                     -dag[cells]*grad[:,2,None])/4
            dsource+=(_mean(ap,cells)[:,None]*(pg[:,0,None]*G[:,:,0]+pg[:,1,None]*G[:,:,1])
                      +(_mean(az,cells)*pg[:,2]-_mean(ag,cells))[:,None]*G[:,:,2])
            Gp=G[:,:,:2]
            K=(_mean(hz,cells)[:,None,None]*np.einsum('eik,ejk->eij',Gp,Gp)
               +_mean(vz,cells)[:,None,None]*G[:,:,None,2]*G[:,None,:,2])
            fu_p=np.einsum('ek,eik->ei',grad[:,:2],Gp)
            fu_z=grad[:,2,None]*G[:,:,2]
            K+=fu_p[:,:,None]*dhz[cells][:,None,:]/4+fu_z[:,:,None]*dvz[cells][:,None,:]/4
            return tau*(vol[:,None,None]*K+(vol/4)[:,None,None]*dsource[:,None,:])
        local=np.concatenate(self._ordered_tensors(tensor),axis=0)
        rows=np.broadcast_to(self.t[:,:,None],local.shape).ravel()
        cols=np.broadcast_to(self.t[:,None,:],local.shape).ravel()
        return coo_matrix((local.ravel(),(rows,cols)),shape=(self.n,self.n)).tocsr()+diags(self.m)
