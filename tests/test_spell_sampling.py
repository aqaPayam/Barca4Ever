import unittest
import numpy as np
from spell_sampling import spell_refs


class SpellSamplingTests(unittest.TestCase):
    def test_brief_spells_get_exposure_and_unknown_is_excluded(self):
        y=np.r_[np.zeros(1000),np.ones(3),np.zeros(1000),np.full(100,3)].astype(np.int8)
        refs=spell_refs([dict(y=y)],9000,np.random.default_rng(51))
        targets=refs[:,1]
        self.assertTrue((targets>=47).all())
        self.assertTrue((targets<len(y)-16).all())
        self.assertTrue((y[targets]<2).all())
        self.assertGreater(((targets>=1000)&(targets<1003)).mean(),.25)


if __name__=='__main__':unittest.main()
