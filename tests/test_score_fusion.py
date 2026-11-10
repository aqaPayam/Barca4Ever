import unittest
import numpy as np
from smart_fusion import reliability
from score_fusion import combine_score


class ScoreFusionTests(unittest.TestCase):
    def test_equal_baseline_probability_bounds_and_team_exchange(self):
        rng=np.random.default_rng(114)
        t=rng.dirichlet([1,1,1],90);n=rng.dirichlet([1,1,1],90)
        x=np.zeros((90,59));x[:,7]=np.arange(90)%2
        f=reliability(t,n,x)
        np.testing.assert_array_equal(combine_score(dict(a=0,b=0),t,n,f),.5*(t+n))
        for a,b in [(-.2,.2),(.2,-.2),(.2,.2),(-.2,-.2)]:
            p=combine_score(dict(a=a,b=b),t,n,f)
            np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
            self.assertTrue(np.all(p>=0))
            swapped=combine_score(dict(a=a,b=b),t[:,[1,0,2]],n[:,[1,0,2]],reliability(t[:,[1,0,2]],n[:,[1,0,2]],x))
            np.testing.assert_allclose(swapped,p[:,[1,0,2]],atol=1e-10)


if __name__=='__main__':unittest.main()
