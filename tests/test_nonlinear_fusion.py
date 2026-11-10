import io
import unittest
import numpy as np
import learned_possession
import joblib
from smart_fusion import reliability
from nonlinear_fusion import fit_regret,combine_regret


class NonlinearFusionTests(unittest.TestCase):
    def test_learns_reliability_and_survives_serialization(self):
        count=10000;y=np.zeros(count,dtype=int)
        good=np.tile([.8,.1,.1],(count,1));bad=np.tile([.1,.8,.1],(count,1))
        t=good.copy();n=bad.copy();t[count//2:]=bad[count//2:];n[count//2:]=good[count//2:]
        f=np.repeat([-1.,1.],count//2)[:,None]
        model=fit_regret([dict(y=y,tree=t,neural=n,f=f)])
        p=combine_regret(model,.4,t,n,f)
        self.assertGreater(float(p[:,0].mean()),.58)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
        np.testing.assert_array_equal(combine_regret(joblib.load(stream),.4,t,n,f),p)


if __name__=='__main__':unittest.main()
