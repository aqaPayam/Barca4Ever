import unittest
import numpy as np
from touch_physics import features, FEATURE_NAMES, TEAM_SWAP_INDEX


class TouchPhysicsTests(unittest.TestCase):
    def scene(self, n=81, fps=25):
        t=(np.arange(n)-n//2)/fps
        players=np.zeros((n,4,2))
        players[:,0,0]=4*t
        players[:,1,0]=12
        players[:,2,0]=20
        players[:,3,0]=30
        return players,np.array([0,0,1,1]),np.zeros((n,2)),np.array([105,68])

    def test_feature_contract_and_exact_contact_valley(self):
        p,teams,b,pitch=self.scene()
        x=features(p,teams,b,pitch)
        self.assertEqual(x.shape,(81,30))
        self.assertEqual(len(set(FEATURE_NAMES)),30)
        self.assertEqual(x.dtype,np.float32)
        for block,lag in enumerate([2,6,12]):
            start=block*7
            self.assertAlmostEqual(float(x[40,start]),4*lag/25/10,places=6)
            self.assertAlmostEqual(float(x[40,start+1]),.4,places=6)
            self.assertAlmostEqual(float(x[40,start+2]),.4,places=6)
            np.testing.assert_array_equal(x[40,start+3:start+6],0)
            self.assertEqual(x[40,start+6],1)
        self.assertEqual(x[40,23],1)
        self.assertEqual(x[40,24],0)  # stationary ball has no turn direction

    def test_team_swap_and_slot_permutation(self):
        p,teams,b,pitch=self.scene()
        x=features(p,teams,b,pitch)
        permutation=[3,0,2,1]
        np.testing.assert_array_equal(features(p[:,permutation],teams[permutation],b,pitch),x)
        np.testing.assert_array_equal(features(p,1-teams,b,pitch),x[:,TEAM_SWAP_INDEX])
        np.testing.assert_array_equal(TEAM_SWAP_INDEX[TEAM_SWAP_INDEX],np.arange(30))

    def test_no_silent_interpolation_across_internal_ball_gap(self):
        p,teams,b,pitch=self.scene()
        b[37]=np.nan
        x=features(p,teams,b,pitch)
        self.assertEqual(x[40,6],1)  # +/-2f does not contain the gap
        self.assertEqual(x[40,13],0)
        self.assertEqual(x[40,20],0)
        self.assertEqual(x[40,23],0)
        self.assertEqual(x[40,24],0)
        np.testing.assert_array_equal(x[40,7:13],0)
        self.assertAlmostEqual(float(x[37,28]),1/25,places=7)
        self.assertAlmostEqual(float(x[37,29]),1/25,places=7)
        self.assertEqual(x[37,25],0)

    def test_correct_physical_velocity_and_acceleration(self):
        p,teams,b,pitch=self.scene()
        t=(np.arange(len(b))-40)/25
        b[:,0]=.5*2*t*t+10*t
        x=features(p,teams,b,pitch)
        self.assertAlmostEqual(float(x[40,21]),2/30,places=6)
        self.assertAlmostEqual(float(x[40,22]),0,places=6)
        self.assertEqual(x[40,24],1)
        b[:,0]=10*t
        x=features(p,teams,b,pitch)
        self.assertAlmostEqual(float(x[40,21]),0,places=6)

    def test_direction_reversal(self):
        p,teams,b,pitch=self.scene()
        b[:,0]=abs((np.arange(len(b))-40)/25)
        x=features(p,teams,b,pitch)
        self.assertAlmostEqual(float(x[40,22]),1,places=6)
        self.assertEqual(x[40,24],1)

    def test_physical_windows_scale_with_fps(self):
        p25,teams,b25,pitch=self.scene(101,25)
        p50,_,b50,_=self.scene(201,50)
        np.testing.assert_allclose(features(p25,teams,b25,pitch,25)[50],
            features(p50,teams,b50,pitch,50)[100],atol=1e-7)

    def test_all_missing_empty_and_absent_team(self):
        p,teams,b,pitch=self.scene()
        all_missing=features(np.full_like(p,np.nan),teams,np.full_like(b,np.nan),pitch)
        self.assertTrue(np.isfinite(all_missing).all())
        np.testing.assert_array_equal(all_missing[:,:28],0)
        np.testing.assert_array_equal(all_missing[:,28:],10)
        self.assertEqual(features(p[:0],teams,b[:0],pitch).shape,(0,30))
        x=features(p[:,:2],teams[:2],b,pitch)
        self.assertEqual(x[40,26],1)
        self.assertEqual(x[40,27],0)
        self.assertEqual(x[40,13],0)
        self.assertEqual(x[40,23],1)  # ball dynamics need no visible player
        self.assertTrue(np.isfinite(features(p[:,:0],teams[:0],b,pitch)).all())

    def test_extreme_coordinates_are_guarded(self):
        p,teams,b,pitch=self.scene()
        p[10]=1e300;b[20]=[-1e300,1e300]
        x=features(p,teams,b,pitch)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(abs(x[:,:21])<=5))
        self.assertTrue(np.all(x[:,21]<=10))

    def test_input_validation(self):
        p,teams,b,pitch=self.scene()
        for args in [(p,teams,b[:2],pitch,25),(p,teams+2,b,pitch,25),
                     (p,teams,b,[-1,68],25),(p,teams,b,pitch,0)]:
            with self.assertRaises(ValueError):features(*args)


if __name__=='__main__':unittest.main()
