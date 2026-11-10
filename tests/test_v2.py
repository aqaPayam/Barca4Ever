import unittest
import numpy as np
from possession_v2 import features, interpolate_short_gaps, spell_weights


class V2Tests(unittest.TestCase):
    def test_interpolation_only_inside_short_observed_gap(self):
        ball=np.array([[np.nan,np.nan],[0.,0.],[np.nan,np.nan],[2.,2.],[np.nan,np.nan]])
        out=interpolate_short_gaps(ball,fps=1,max_gap_s=1)
        np.testing.assert_array_equal(out[2],[1,1])
        self.assertTrue(np.isnan(out[[0,4]]).all())
        self.assertTrue(np.isnan(ball[2]).all())

    def test_long_gaps_stay_unknown(self):
        ball=np.full((100,2),np.nan); ball[0]=0; ball[-1]=1
        self.assertTrue(np.isnan(interpolate_short_gaps(ball)[1:-1]).all())

    def test_short_spells_have_more_loss_weight(self):
        y=np.array([0]*5+[1]*500+[2]*100)
        w=spell_weights(y)
        self.assertGreater(w[0],w[10])
        np.testing.assert_array_equal(w[-100:],np.ones(100))

    def test_features_all_missing_still_finite(self):
        p=np.full((100,6,2),np.nan); b=np.full((100,2),np.nan)
        x=features(p,np.array([0,0,0,1,1,1]),b,(105,68))
        self.assertEqual(x.shape,(100,59))
        self.assertTrue(np.isfinite(x).all())

    def test_permuting_slots_preserves_features(self):
        rng=np.random.default_rng(5)
        p=rng.normal(size=(100,6,2)).astype(np.float32)
        b=rng.normal(size=(100,2)).astype(np.float32)
        teams=np.array([0,0,0,1,1,1]); ix=[5,1,3,2,4,0]
        np.testing.assert_allclose(features(p,teams,b,(105,68)),
            features(p[:,ix],teams[ix],b,(105,68)),atol=1e-5)


if __name__=='__main__': unittest.main()
