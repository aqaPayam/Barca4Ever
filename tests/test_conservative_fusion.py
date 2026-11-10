import unittest
import numpy as np
from smart_fusion import reliability,combine
from conservative_fusion import blend


class ConservativeFusionTests(unittest.TestCase):
    def test_convex_probabilities_and_team_symmetry(self):
        rng=np.random.default_rng(19)
        t=rng.dirichlet([1,1,1],80);n=rng.dirichlet([1,1,1],80)
        x=np.zeros((80,59));f=reliability(t,n,x);onset=rng.random((80,3))
        t[0]=[.5,.5,0];n[0]=[.7,.3,0]
        f=reliability(t,n,x)
        def head(sign):return dict(center=np.zeros(f.shape[1]),scale=np.ones(f.shape[1]),beta=np.r_[sign,np.zeros(f.shape[1])])
        for kind in ['joint','factorized']:
            gate=dict(kind=kind,heads=[head(1)] if kind=='joint' else [head(1),head(-1)])
            base=.5*(t+n);learned=combine(gate,t,n,f)
            for variant in ['shrink25','shrink50','disagree','boundary']:
                p=blend(gate,variant,t,n,f,onset)
                np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
                self.assertTrue(np.all(p>=np.minimum(base,learned)-1e-12))
                self.assertTrue(np.all(p<=np.maximum(base,learned)+1e-12))
                order=[1,0,2]
                swapped=blend(gate,variant,t[:,order],n[:,order],reliability(t[:,order],n[:,order],x),onset[:,order])
                np.testing.assert_allclose(swapped,p[:,order],atol=1e-10)
                if variant in ['disagree','boundary']:
                    np.testing.assert_allclose(p[0],base[0])


if __name__=='__main__':unittest.main()
