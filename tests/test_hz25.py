import unittest
import numpy as np
from hz25_data import object_features


class ObservabilityInputsTests(unittest.TestCase):
    def test_actual_elapsed_time_preserves_constant_speed(self):
        n=125;t=np.arange(n)/25
        players=np.zeros((n,2,2),np.float32);players[:,0,0]=2*t;players[:,1,0]=-2*t
        ball=np.column_stack([3*t,np.zeros(n)]).astype(np.float32)
        for rate in [16,25]:
            x,s=object_features(players,np.array([0,1]),ball,[105,68],rate)
            np.testing.assert_allclose(x[1:,2,2],2/52.5,atol=1e-6)
            np.testing.assert_allclose(x[1:,0,2],3/52.5,atol=1e-6)
            self.assertTrue(np.isfinite(x).all())

    def test_raw_visibility_survives_interpolation(self):
        p=np.zeros((100,2,2),np.float32);b=np.zeros((100,2),np.float32);b[10:35]=np.nan
        for rate in [16,25]:
            x,s=object_features(p,np.array([0,1]),b,[105,68],rate)
            gap=(s>=10)&(s<35)
            self.assertTrue((x[gap,0,17]==0).all())
            self.assertTrue((x[gap,0,10]==0).all())
            self.assertTrue((x[gap,0,18:20]>0).all())
            self.assertTrue((x[~gap,0,17]==1).all())

    def test_all_missing_is_finite_and_explicit(self):
        p=np.full((70,2,2),np.nan,np.float32);b=np.full((70,2),np.nan,np.float32)
        x,s=object_features(p,np.array([0,1]),b,[105,68],25)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue((x[:,:,:2]==-2).all());self.assertTrue((x[:,:,17]==0).all())


class ExpandedNetworkTests(unittest.TestCase):
    def test_zero_extra_weights_preserve_old_function(self):
        from torch_runtime import torch
        from paper2023_model import PaperPossession
        from hz25_model import expand
        torch.set_num_threads(2);torch.manual_seed(35)
        old=PaperPossession(channels=17,instant_skip=True,geometry_attention=True).eval()
        architecture=dict(width=64,dropout=.1,channels=17,instant_skip=True,geometry_attention=True)
        new,a=expand(dict(architecture=architecture,state_dict=old.state_dict()))
        new.eval();x=torch.randn(2,64,36,17);x[:,:,:,10]=0;x[:,:,:,16]=1
        augmented=torch.cat([x,torch.randn(2,64,36,3)],dim=-1)
        with torch.inference_mode():
            expected=old(x);actual=new(augmented)
        for e,v in zip(expected,actual):torch.testing.assert_close(e,v,atol=1e-6,rtol=1e-6)
        self.assertTrue((new.encoder.stem.weight[:,17:]==0).all())
        self.assertTrue((new.instant[0].weight[:,17:]==0).all())


if __name__=='__main__':unittest.main()
