import unittest
import numpy as np
from tta_features import team_swap,reflect_pitch,restore_team_probabilities,average_predictions
from paper2023_data import relative_features


class TTATests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(71)
        self.raw=rng.normal(size=(90,36,11)).astype(np.float32)
        self.raw[:,:,10]=0;self.raw[:,:,6]=0
        self.raw[:,:,4:10]=0;self.raw[:,0,4]=1;self.raw[:,1,5]=1
        self.raw[:,2:19,7]=1;self.raw[:,19:36,8]=1
        self.raw[:,1,:4]=-2;self.raw[:,1,10]=1
        self.raw[20:30,8,:4]=-2;self.raw[20:30,8,10]=1
        self.raw[30,8,2:4]=-2

    def test_team_exchange_and_probability_label_inverse(self):
        np.testing.assert_array_equal(team_swap(team_swap(self.raw)),self.raw)
        swapped=team_swap(self.raw)
        np.testing.assert_array_equal(swapped[:,2,:4],self.raw[:,19,:4])
        self.assertTrue((swapped[:,2,7]==1).all())
        self.assertTrue((swapped[:,2,8]==0).all())
        p=np.array([[.1,.7,.2],[.5,.2,.3]],np.float32)
        np.testing.assert_array_equal(restore_team_probabilities(restore_team_probabilities(p)),p)
        np.testing.assert_array_equal(restore_team_probabilities(p),p[:,[1,0,2]])

    def test_reflection_inverse_and_missing_padding(self):
        reflected=reflect_pitch(self.raw)
        np.testing.assert_array_equal(reflect_pitch(reflected),self.raw)
        np.testing.assert_array_equal(reflected[0,0,:2],-self.raw[0,0,:2])
        np.testing.assert_array_equal(reflected[1,0,2:4],-self.raw[1,0,2:4])
        np.testing.assert_array_equal(reflected[:,1,:4],self.raw[:,1,:4])
        np.testing.assert_array_equal(reflected[20:30,8,:4],self.raw[20:30,8,:4])
        np.testing.assert_array_equal(reflected[30,8,2:4],self.raw[30,8,2:4])
        r=relative_features(reflected)
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue((r[:,1,11:]==0).all())
        self.assertTrue((r[:,1,:4]==-2).all())

    def test_joint_probability_averages_are_normalized(self):
        rng=np.random.default_rng(31)
        paths={key:rng.dirichlet([1,1,1],100) for key in ['original','swap','reflection']}
        for candidate in ['swap','reflection','both']:
            p=average_predictions(paths,candidate)
            np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
            self.assertTrue((p>=0).all())
        np.testing.assert_allclose(average_predictions(paths,'both'),sum(paths.values())/3,atol=1e-12)


if __name__=='__main__':unittest.main()
