import unittest
import numpy as np
from research_trees import boundary_targets,adaptive_decode
from learned_possession import viterbi


class BoundaryTests(unittest.TestCase):
    def test_no_boundary_in_constant_sequence(self):
        np.testing.assert_array_equal(boundary_targets(np.zeros(50)),np.zeros(50))
    def test_boundary_window_is_centered(self):
        y=np.r_[np.zeros(10),np.ones(10)]
        self.assertEqual(np.flatnonzero(boundary_targets(y)).tolist(),list(range(7,14)))
    def test_zero_scale_equals_original_viterbi(self):
        p=np.random.default_rng(3).dirichlet([1,1,1],200)
        np.testing.assert_array_equal(adaptive_decode(p,np.ones(200),8,0),viterbi(p,8))
    def test_boundary_can_enable_brief_control(self):
        p=np.tile([.9,.09,.01],(40,1));p[18:22]=[.15,.84,.01]
        low=adaptive_decode(p,np.zeros(40),8,16)
        high=adaptive_decode(p,np.ones(40),8,16)
        self.assertTrue((low[18:22]==0).all())
        self.assertTrue((high[18:22]==1).all())
    def test_production_decoder_matches_research(self):
        from temporal_decoders import adaptive_decode as production,boundary_features as production_features
        from research_trees import boundary_features
        rng=np.random.default_rng(9)
        p=rng.dirichlet([1,1,1],300);b=rng.random(300);x=rng.normal(size=(300,59))
        np.testing.assert_array_equal(production(p,b,16,4),adaptive_decode(p,b,16,4))
        np.testing.assert_array_equal(production_features(x),boundary_features(x))


class TCNTests(unittest.TestCase):
    def setUp(self):
        try:
            import research_tcn
        except ImportError:
            self.skipTest('Optional PyTorch runtime unavailable in this execution context')
        self.module=research_tcn
        self.module.torch.set_num_threads(2)
    def test_output_shapes_preserve_frame_resolution(self):
        m=self.module
        a,b=m.PossessionTCN()(m.torch.zeros(2,59,128))
        self.assertEqual(tuple(a.shape),(2,3,128));self.assertEqual(tuple(b.shape),(2,128))
    def test_chunk_context_matches_whole_sequence(self):
        m=self.module;m.torch.manual_seed(4)
        model=m.PossessionTCN().eval()
        x=np.random.default_rng(4).normal(size=(4300,59)).astype(np.float32)
        p,_=m.probabilities(model,x,np.zeros(59),np.ones(59))
        with m.torch.no_grad():
            a,_=model(m.torch.from_numpy(x.T.copy())[None])
            full=a.softmax(1)[0].T.numpy()
        np.testing.assert_allclose(p,full,atol=1e-6)


if __name__=='__main__':unittest.main()
