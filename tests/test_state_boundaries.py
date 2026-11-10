import itertools
import unittest
import numpy as np
from research_state_boundaries import decode,onset_targets


class StateBoundaryTests(unittest.TestCase):
    def test_targets_identify_incoming_state(self):
        y=np.r_[np.zeros(10),np.ones(10),np.full(10,2)]
        target=onset_targets(y)
        self.assertEqual(target[10],2)  # entering away, not leaving home
        self.assertEqual(target[20],3)  # entering dead
        self.assertEqual(target[0],0)
    def test_decoder_maximizes_destination_cost_objective(self):
        rng=np.random.default_rng(51)
        p=rng.dirichlet([2,2,2],5);b=rng.random((5,3))
        cost=8/(1+4*b)
        def objective(sequence):
            return sum(np.log(p[i,k]) for i,k in enumerate(sequence))-sum(
                cost[i,sequence[i]] for i in range(1,len(sequence)) if sequence[i]!=sequence[i-1])
        best=max(itertools.product(range(3),repeat=5),key=objective)
        self.assertEqual(tuple(decode(p,b,8,4)),best)


if __name__=='__main__':unittest.main()
