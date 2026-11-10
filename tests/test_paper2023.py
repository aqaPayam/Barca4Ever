"""Architecture and data-contract checks using synthetic observations only."""
import unittest
import numpy as np
from paper2023_data import object_features,relative_features


class ObjectInputTests(unittest.TestCase):
    def test_roles_missingness_and_pitch_scaling(self):
        players=np.zeros((100,2,2),np.float32);players[:,0,0]=52.5
        ball=np.zeros((100,2),np.float32)
        x,sample=object_features(players,np.array([0,1]),ball,[105,68],[True,False])
        self.assertEqual(x.shape,(64,36,11))
        self.assertEqual(sample[0],0)
        self.assertEqual(x[0,0,4],1)  # ball flag
        self.assertEqual(x[0,1,5],1)  # missing referee flag
        self.assertEqual(x[0,1,10],1)
        self.assertEqual(x[0,1,0],-2)
        self.assertEqual(x[0,2,0],1)  # half-pitch length -> one
        self.assertEqual(x[0,2,6],1)  # goalkeeper
        self.assertEqual(x[0,2,7],1)  # home
        self.assertEqual(x[0,19,8],1)  # away
    def test_all_missing_is_finite_and_has_no_fake_ball(self):
        p=np.full((100,2,2),np.nan);b=np.full((100,2),np.nan)
        x,_=object_features(p,np.array([0,1]),b,[105,68],[False,False])
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue((x[:,:,10]==1).all())
        self.assertTrue((x[:,:,:2]==-2).all())
    def test_relative_features_preserve_missing_pairs(self):
        p=np.zeros((100,2,2),np.float32);b=np.zeros((100,2),np.float32)
        p[:,0,0]=10
        x,_=object_features(p,np.array([0,1]),b,[105,68],[False,False])
        r=relative_features(x)
        self.assertEqual(r.shape[-1],17)
        np.testing.assert_allclose(r[:,2,11],1,atol=1e-6)
        self.assertTrue((r[:,1,11:]==0).all())  # no fabricated referee-ball distance
    def test_relative_velocity_scaling_keeps_missing_sentinel(self):
        x=np.zeros((3,36,11),np.float32)
        x[:,:,2:4]=[5.25,3.4]
        x[:,1,10]=1
        r=relative_features(x)
        np.testing.assert_allclose(r[:,2,2:4],.1,atol=1e-6)
        self.assertTrue((r[:,1,2:4]==-2).all())
    def test_contact_adapter_keeps_more_than_seventeen_slots(self):
        from contact_input import contact_features
        p=np.zeros((100,40,2),np.float32);b=np.zeros((100,2),np.float32)
        teams=np.r_[np.zeros(20),np.ones(20)].astype(int)
        x,_=contact_features(p,teams,b,[105,68],np.zeros(40,bool))
        self.assertEqual(x.shape,(64,70,17))
        self.assertEqual(int((x[0,:,10]==0).sum()),41)  # all forty players plus ball
    def test_goalkeeper_metadata_supports_roles_and_safe_missing_fallback(self):
        from io import StringIO
        from contact_input import goalkeeper_flags
        meta=StringIO('<root><Role id="g" name="Goalkeeper"/><Player roleId="g"/>'
            '<Player><Position>GK</Position></Player><Player isGoalkeeper="true"/>'
            '<Player name="GK" role="defender"/></root>')
        self.assertEqual(goalkeeper_flags(meta,4).tolist(),[True,True,True,False])
    def test_geometry_role_fallback_uses_both_pitch_ends(self):
        from contact_input import infer_goalkeepers
        p=np.tile(np.array([[-48,0],[-20,10],[48,0],[20,10]],np.float32),(40,1,1))
        flags=infer_goalkeepers(p,np.array([0,0,1,1]),[105,68])
        self.assertEqual(flags.tolist(),[True,False,True,False])


class PaperArchitectureTests(unittest.TestCase):
    def setUp(self):
        try:
            import paper2023_model as module
        except ImportError:
            self.skipTest('Optional PyTorch runtime unavailable')
        self.m=module;self.t=module.torch
        self.t.set_num_threads(2);self.t.manual_seed(31)
        self.model=module.PaperPossession(width=8,dropout=0).eval()
    def test_player_order_does_not_change_prediction(self):
        x=self.t.randn(2,64,36,11)
        permutation=np.r_[0,1,np.random.default_rng(2).permutation(np.arange(2,36))]
        with self.t.no_grad():
            a,b=self.model(x);c,d=self.model(x[:,:,permutation])
        self.t.testing.assert_close(a,c,atol=1e-6,rtol=1e-6)
        self.t.testing.assert_close(b,d,atol=1e-6,rtol=1e-6)
    def test_dead_frames_give_zero_home_gradient(self):
        dead=self.t.zeros(10,requires_grad=True);home=self.t.zeros(10,requires_grad=True)
        loss=self.m.joint_loss(dead,home,self.t.full((10,),2,dtype=self.t.long))
        loss.backward()
        self.assertTrue((home.grad==0).all())
        self.assertTrue((dead.grad!=0).all())
    def test_dense_predictions_match_direct_64_frame_windows(self):
        x=np.random.default_rng(3).normal(size=(113,36,11)).astype(np.float32)
        dense=self.m.outputs(self.model,x,'cpu',chunk=31)
        targets=np.array([0,1,31,48,63,112])
        windows=np.stack([x[np.clip(np.arange(i-47,i+17),0,len(x)-1)] for i in targets])
        with self.t.no_grad():
            a,b=self.model(self.t.from_numpy(windows))
        direct=np.column_stack([a.sigmoid().numpy(),b.sigmoid().numpy()])
        np.testing.assert_allclose(dense[targets],direct,atol=1e-6)
    def test_last_input_is_sixteen_steps_after_target(self):
        # A known index in the target's window must be index 47; last is t+16.
        target=100;window=np.arange(target-47,target+17)
        self.assertEqual(window[47],target)
        self.assertEqual(window[-1],target+16)
    def test_joint_probabilities_sum_to_one(self):
        p=self.m.joint_probabilities(np.array([[.2,.3],[.9,.1]],dtype=np.float32))
        np.testing.assert_allclose(p.sum(1),1)
    def test_contact_skip_dense_predictions_keep_target_alignment(self):
        model=self.m.PaperPossession(width=8,channels=17,instant_skip=True,geometry_attention=True,dropout=0).eval()
        x=np.random.default_rng(8).normal(size=(100,36,17)).astype(np.float32)
        dense=self.m.outputs(model,x,'cpu',chunk=27)
        targets=[0,40,99]
        windows=np.stack([x[np.clip(np.arange(i-47,i+17),0,len(x)-1)] for i in targets])
        with self.t.no_grad():a,b=model(self.t.from_numpy(windows))
        np.testing.assert_allclose(dense[targets],np.column_stack([a.sigmoid().numpy(),b.sigmoid().numpy()]),atol=1e-6)
    def test_geometry_prior_prefers_nearby_observed_players(self):
        attention=self.m.BallAttention(8,geometry_attention=True)
        with self.t.no_grad():
            attention.query.weight.zero_();attention.query.bias.zero_()
            attention.key.weight.zero_();attention.key.bias.zero_()
            attention.value.weight.copy_(self.t.eye(8));attention.value.bias.zero_()
            z=self.t.zeros(1,4,8);z[0,2,0]=1
            g=self.t.zeros(1,4,17);g[:,:,16]=1;g[0,2,15]=.1;g[0,3,15]=5
            result=attention(z,g,players_only=True)
        self.assertGreater(float(result[0,0]),.99)
    def test_extra_missing_slots_do_not_change_contact_prediction(self):
        model=self.m.PaperPossession(width=8,channels=17,instant_skip=True,geometry_attention=True,dropout=0).eval()
        x=self.t.randn(2,64,36,17);x[:,:,:,10]=0;x[:,:,:,16]=1
        pad=self.t.zeros(2,64,34,17);pad[:,:,:,:4]=-2;pad[:,:,:,10]=1
        with self.t.no_grad():
            a,b=model(x);c,d=model(self.t.cat([x,pad],dim=2))
        self.t.testing.assert_close(a,c,atol=1e-6,rtol=1e-6)
        self.t.testing.assert_close(b,d,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
