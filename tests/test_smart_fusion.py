import unittest
import numpy as np
from smart_fusion import reliability,objective,fit_gate,combine


class SmartFusionTests(unittest.TestCase):
    def test_analytic_gradient(self):
        rng=np.random.default_rng(73)
        z=rng.normal(size=(60,5));beta=rng.normal(0,.2,5)
        t=rng.uniform(.05,.8,60);n=rng.uniform(.05,.8,60);w=rng.uniform(.2,3,60)
        _,gradient=objective(beta,z,t,n,w)
        numerical=[]
        for i in range(5):
            delta=np.eye(5)[i]*1e-6
            numerical.append((objective(beta+delta,z,t,n,w)[0]-objective(beta-delta,z,t,n,w)[0])/2e-6)
        np.testing.assert_allclose(gradient,numerical,atol=1e-8)

    def test_probability_and_team_exchange_invariance(self):
        rng=np.random.default_rng(97)
        t=rng.dirichlet([1,1,1],300);n=rng.dirichlet([1,1,1],300);x=rng.uniform(0,4,(300,59))
        y=rng.integers(0,3,300)
        f=reliability(t,n,x)
        swapped=x.copy()
        for a,b in [(0,3),(1,4),(2,5),(8,9),(42,45),(43,46),(44,47)]:swapped[:,[a,b]]=swapped[:,[b,a]]
        swapped[:,6]*=-1;swapped[:,48]*=-1
        fs=reliability(t[:,[1,0,2]],n[:,[1,0,2]],swapped)
        np.testing.assert_allclose(f,fs,atol=1e-12)
        for kind in ['joint','factorized']:
            gate=fit_gate([dict(y=y,tree=t,neural=n,f=f)],kind)
            p=combine(gate,t,n,f)
            np.testing.assert_allclose(p.sum(1),1,atol=1e-8)
            np.testing.assert_allclose(combine(gate,t[:,[1,0,2]],n[:,[1,0,2]],fs),p[:,[1,0,2]],atol=1e-10)
            extreme=np.zeros((300,3));extreme[:,2]=1
            np.testing.assert_allclose(combine(gate,extreme,extreme,f).sum(1),1,atol=1e-8)

    def test_gate_learns_a_reliability_signal(self):
        rng=np.random.default_rng(23);y=np.zeros(600,dtype=int)
        indicator=np.repeat([-1.,1.],300)
        t=np.tile([.8,.1,.1],(600,1));n=t.copy()
        t[indicator>0]=[.1,.8,.1];n[indicator<0]=[.1,.8,.1]
        f=indicator[:,None]
        gate=fit_gate([dict(y=y,tree=t,neural=n,f=f)],'joint')
        p=combine(gate,t,n,f)
        self.assertGreater(float(p[:,0].mean()),.6)


if __name__=='__main__':unittest.main()
