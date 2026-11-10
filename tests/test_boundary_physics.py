import unittest
import numpy as np
from boundary_physics import physics_features,combined_features,dense_sampling,PHYSICS_NAMES
from boundary_refinement import targets


class DenseBoundaryPhysicsTests(unittest.TestCase):
    def test_added_missing_slots_do_not_change_selected_physics(self):
        n=80;time=np.arange(n)/25
        ball=np.column_stack([time,np.sin(time)])
        players=np.zeros((n,4,2));players[:,:,0]=np.array([-2,1,3,5])+time[:,None]
        teams=np.array([0,0,1,1]);ball[30:33]=np.nan
        expected=physics_features(players,teams,ball,[105,68])
        padded=np.concatenate([players,np.full((n,16,2),np.nan)],axis=1)
        tt=np.r_[teams,np.tile([0,1],8)]
        np.testing.assert_allclose(expected,physics_features(padded,tt,ball,[105,68]),atol=1e-7)
        self.assertEqual(expected.shape,(n,28))
        self.assertFalse(any('fraction' in name for name in PHYSICS_NAMES))
        self.assertEqual(combined_features(np.zeros((n,59)),expected).shape,(n,321))

    def test_tiny_changes_and_nearby_negatives_are_preserved(self):
        y=np.zeros(60,dtype=int);y[10]=1;y[40:45]=3
        take,w=dense_sampling(y);t=targets(y)
        self.assertTrue(take[10] and take[11])
        self.assertNotEqual(t[10],t[11])
        self.assertTrue(np.all(w[t>0]==4))
        self.assertEqual(w[0],.2)
        self.assertEqual(w[25],1)
        self.assertTrue(take[25]);self.assertFalse(take[26])
        self.assertFalse(take[40:45].any())


if __name__=='__main__':unittest.main()
