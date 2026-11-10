import unittest
import numpy as np
from torch_runtime import torch
from paper2023_model import PaperPossession,joint_loss
from residual_missing_data import append_missing
from residual_missing_model import initialize,restore_protected_columns,protected_proof,INPUT_WEIGHTS,extra_norm


class ResidualMissingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_extra_channels_zero_on_observed_and_first17_unchanged(self):
        base=np.random.default_rng(8).normal(size=(80,36,17)).astype(np.float32)
        samples=np.rint(np.arange(80)*25/16).astype(int)
        ball=np.zeros((125,2),np.float32)
        observed=append_missing(base,ball,samples)
        self.assertEqual(observed[:,:,:17].tobytes(),base.tobytes())
        self.assertTrue((observed[:,:,17:]==0).all())
        ball[40:60]=np.nan
        missing=append_missing(base,ball,samples)
        take=(samples>=40)&(samples<60)
        self.assertTrue((missing[take,:,17]==1).all())
        self.assertTrue((missing[take,:,18:]>0).all())

    def test_protected_tensors_and_gradients_survive_optimizer_step(self):
        torch.manual_seed(19)
        architecture=dict(width=8,channels=17,dropout=.1,instant_skip=True,geometry_attention=True)
        legacy=PaperPossession(**architecture).eval()
        parent=dict(architecture=architecture,state_dict={k:v.detach().clone() for k,v in legacy.state_dict().items()})
        model,_,protected,handles=initialize(parent,'cpu')
        parameters=dict(model.named_parameters())
        self.assertEqual({name for name,p in parameters.items() if p.requires_grad},set(INPUT_WEIGHTS))
        x=torch.randn(8,64,36,20);x[:,:,:,6]=0;x[:,:,:,10]=0;x[:,:,:,16]=1
        x[:,:,:,17:]=torch.rand(8,64,36,3)
        dead,home=model(x);loss=joint_loss(dead,home,torch.tensor([0,1,2,0,1,2,0,1]))
        loss.backward()
        for name in INPUT_WEIGHTS:self.assertTrue((parameters[name].grad[:,:17]==0).all())
        optimizer=torch.optim.Adam([parameters[n] for n in INPUT_WEIGHTS],lr=.0005,weight_decay=0)
        optimizer.step();restore_protected_columns(model,protected)
        proof=protected_proof(model,parent)
        self.assertEqual(proof['before_sha256'],proof['after_sha256'])
        self.assertGreater(extra_norm(model),0)
        self.assertTrue(all(not module.training for module in model.modules()))

    def test_fully_observed_window_preserves_legacy_function_after_learning(self):
        torch.manual_seed(24)
        architecture=dict(width=8,channels=17,dropout=.1,instant_skip=True,geometry_attention=True)
        legacy=PaperPossession(**architecture).eval()
        parent=dict(architecture=architecture,state_dict={k:v.detach().clone() for k,v in legacy.state_dict().items()})
        model,_,protected,handles=initialize(parent,'cpu')
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name in INPUT_WEIGHTS:p[:,17:].normal_(0,.1)
        restore_protected_columns(model,protected);protected_proof(model,parent)
        x=torch.randn(3,64,36,17);x[:,:,:,6]=0;x[:,:,:,10]=0;x[:,:,:,16]=1
        with torch.inference_mode():
            old=legacy(x);new=model(torch.cat([x,torch.zeros(3,64,36,3)],dim=-1))
        for a,b in zip(old,new):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
