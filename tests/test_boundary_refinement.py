import itertools
import unittest
import numpy as np
from boundary_refinement import temporal_features,targets,PAIRS,decode,blend_scores,matrix_scores
from temporal_state import decode as destination_decode


class BoundaryRefinementTests(unittest.TestCase):
    def test_directed_targets_and_unknowns(self):
        y=np.r_[np.zeros(10),np.ones(10),np.full(10,2),np.zeros(10),np.full(5,3)]
        t=targets(y)
        self.assertEqual(t[10],1)
        self.assertEqual(t[20],4)
        self.assertEqual(t[30],5)
        self.assertTrue(np.all(t[40:]==0))
        self.assertEqual(t[7],0)
        self.assertEqual(t[8],1)

    def test_signed_temporal_features_keep_time_direction(self):
        x=np.broadcast_to(np.arange(40)[:,None],(40,59)).astype(np.float32)
        f=temporal_features(x)
        self.assertEqual(f.shape,(40,293))
        self.assertTrue(np.all(f[20,59+52:59+78]==4))
        reversed_features=temporal_features(x[::-1])
        self.assertTrue(np.all(reversed_features[20,59+52:59+78]==-4))

    def test_target_team_exchange_and_nearest_tie(self):
        y=np.r_[np.zeros(10),np.ones(2),np.zeros(10),np.full(10,2)].astype(int)
        swapped=np.where(y<2,1-y,y)
        mapping=np.array([0,3,4,1,2,6,5])
        np.testing.assert_array_equal(targets(swapped),mapping[targets(y)])
        self.assertEqual(targets(y)[11],1)

    def test_zero_blend_exactly_recovers_v41_decoder(self):
        rng=np.random.default_rng(123)
        p=rng.dirichlet([2,2,2],100);b=rng.random((100,3));pair=rng.random((100,3,3))
        np.testing.assert_array_equal(decode(p,blend_scores(b,pair,0)),destination_decode(p,b,16,8))

    def test_directed_decoder_matches_bruteforce(self):
        rng=np.random.default_rng(51);p=rng.dirichlet([2,2,2],5);b=rng.random((5,3,3))
        def objective(seq):
            return sum(np.log(p[i,k]) for i,k in enumerate(seq))-sum(16/(1+8*b[i,seq[i-1],seq[i]]) for i in range(1,len(seq)) if seq[i]!=seq[i-1])
        expected=max(itertools.product(range(3),repeat=5),key=objective)
        self.assertEqual(tuple(decode(p,b)),expected)

    def test_class_probabilities_route_to_ordered_edges(self):
        class Fake:
            classes_=np.array([0,1,6])
            def predict_proba(self,x):return np.tile([.2,.3,.5],(len(x),1))
        matrix=matrix_scores(Fake(),np.zeros((5,59),np.float32))
        np.testing.assert_allclose(matrix[:,0,1],.3)
        np.testing.assert_allclose(matrix[:,2,1],.5)
        np.testing.assert_array_equal(matrix[:,1,0],0)
        np.testing.assert_array_equal(np.diagonal(matrix,axis1=1,axis2=2),0)


if __name__=='__main__':unittest.main()
