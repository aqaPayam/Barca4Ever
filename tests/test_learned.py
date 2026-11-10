import unittest
import numpy as np
from learned_possession import features, corrupt, viterbi
from infer_learned import tracking_blocks


class LearnedTests(unittest.TestCase):
    def test_features_finite_with_missing_tracking(self):
        p = np.full((80, 6, 2), np.nan, dtype=np.float32)
        b = np.full((80, 2), np.nan, dtype=np.float32)
        x = features(p, np.array([0, 0, 0, 1, 1, 1]), b, (105, 68))
        self.assertEqual(x.shape, (80, 42))
        self.assertTrue(np.isfinite(x).all())

    def test_slot_order_is_irrelevant(self):
        rng = np.random.default_rng(0)
        p = rng.normal(size=(80, 6, 2)).astype(np.float32)
        b = rng.normal(size=(80, 2)).astype(np.float32)
        teams = np.array([0, 0, 0, 1, 1, 1])
        ix = np.array([4, 1, 5, 0, 3, 2])
        np.testing.assert_allclose(features(p, teams, b, (105, 68)),
                                   features(p[:, ix], teams[ix], b, (105, 68)), atol=1e-5)

    def test_corruption_reproducible_and_does_not_mutate(self):
        p = np.zeros((100, 6, 2), dtype=np.float32)
        b = np.zeros((100, 2), dtype=np.float32)
        a = corrupt(p, b, 42)
        z = corrupt(p, b, 42)
        np.testing.assert_array_equal(a[0], z[0])
        np.testing.assert_array_equal(a[1], z[1])
        self.assertTrue((p == 0).all() and (b == 0).all())

    def test_viterbi_suppresses_single_frame_flicker(self):
        prob = np.tile([0.9, 0.05, 0.05], (20, 1))
        prob[10] = [0.1, 0.8, 0.1]
        self.assertEqual(viterbi(prob, 0)[10], 1)
        np.testing.assert_array_equal(viterbi(prob, 3), np.zeros(20))

    def test_tracking_gap_resets_but_short_gaps_do_not(self):
        p = np.zeros((1000, 2, 2))
        p[100:105] = np.nan
        p[300:700] = np.nan
        self.assertEqual(tracking_blocks(p, 25), [(0, 300), (700, 1000)])


if __name__ == "__main__":
    unittest.main()
