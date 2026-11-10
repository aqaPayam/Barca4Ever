import unittest
import numpy as np
from torch_runtime import torch
from paper2023_model import PaperPossession
from refined_infer import confidence_features,restore_network,predict_block
from infer_contact import predict_block as old_predict


class ConstantTree:
    classes_=np.array([0,1,2])
    def predict_proba(self,x):return np.tile([.65,.25,.10],(len(x),1))


class ConstantOnset:
    classes_=np.arange(4)
    def predict_proba(self,x):return np.tile([.7,.1,.1,.1],(len(x),1))


class ConstantPair:
    classes_=np.arange(7)
    def predict_proba(self,x):return np.tile(np.r_[.7,np.full(6,.05)],(len(x),1))


class PhysicsPair(ConstantPair):
    def predict_proba(self,x):
        if x.ndim!=2 or x.shape[1]!=321 or not np.isfinite(x).all():
            raise AssertionError('Expected finite321-column signed/physics input')
        self.input_shape=x.shape
        return super().predict_proba(x)


class RefinedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self,count=44):
        rng=np.random.default_rng(41);frames=125
        players=rng.normal(0,8,(frames,count,2)).astype(np.float32)
        teams=np.r_[np.zeros(count//2),np.ones(count-count//2)].astype(int)
        ball=np.column_stack([np.linspace(-3,3,frames),np.ones(frames)]).astype(np.float32)
        ball[40:55]=np.nan
        return players,teams,ball

    def test_grouping_retains_all_observed_actors_and_confidence(self):
        players,teams,ball=self.fixture()
        for rate in [16,25]:
            x,samples=confidence_features(players,teams,ball,[105,68],rate)
            self.assertEqual(x.shape,(len(samples),70,20))
            self.assertEqual(int((x[0,:,10]==0).sum()),45)
            for team,flag in [(0,7),(1,8)]:
                self.assertEqual(int(((x[0,:,10]==0)&(x[0,:,flag]==1)).sum()),22)
            self.assertTrue((x[:,:,6]==0).all())
            self.assertTrue((x[:,2:36,7]==1).all())
            self.assertTrue((x[:,36:,8]==1).all())
            inside=(samples>=40)&(samples<55)
            self.assertTrue((x[inside,0,10]==0).all())
            self.assertTrue((x[inside,:,17]==0).all())
            self.assertTrue((x[inside,:,18:]>0).all())

    def test_random_network_ignores_extra_missing_groups_and_actor_permutation(self):
        players,teams,ball=self.fixture(40)
        order=np.random.default_rng(17).permutation(40)
        padded=np.concatenate([players,np.full((len(ball),32,2),np.nan,np.float32)],axis=1)
        padded_teams=np.r_[teams,np.zeros(16,dtype=int),np.ones(16,dtype=int)]
        torch.manual_seed(21)
        model=PaperPossession(width=8,channels=20,dropout=0,instant_skip=True,geometry_attention=True).eval()
        for rate in [16,25]:
            x,_=confidence_features(players,teams,ball,[105,68],rate)
            permuted,_=confidence_features(players[:,order],teams[order],ball,[105,68],rate)
            expanded,_=confidence_features(padded,padded_teams,ball,[105,68],rate)
            self.assertEqual(expanded.shape[1],104)
            with torch.inference_mode():
                original=model(torch.from_numpy(x[:64])[None])
                for other in [permuted,expanded]:
                    actual=model(torch.from_numpy(other[:64])[None])
                    for a,b in zip(original,actual):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_restore_legacy_compatibility_and_optional_pair_branch(self):
        players,teams,ball=self.fixture(6)
        torch.manual_seed(8)
        architecture=dict(width=8,channels=17,dropout=0,instant_skip=True,geometry_attention=True)
        model=PaperPossession(**architecture).eval()
        artifact=dict(kind='refined_contact',role_mode='none',input_mode='legacy17',rate=16,
            architecture=architecture,neural_state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()},
            neural_weight=.5,tree_model=ConstantTree(),boundary_model=ConstantOnset(),
            boundary_kind='state',penalty=16,boundary_scale=8)
        restored=restore_network(artifact,'cpu')
        original=old_predict(artifact,model,players,teams,ball,[105,68],device='cpu')
        actual=predict_block(artifact,restored,players,teams,ball,[105,68],device='cpu')
        np.testing.assert_array_equal(actual,original)
        with_pair=dict(artifact,pair_boundary_model=ConstantPair(),pair_alpha=0.)
        np.testing.assert_array_equal(predict_block(with_pair,restored,players,teams,ball,[105,68]),original)
        invalid=dict(artifact,rate=25)
        with self.assertRaises(ValueError):restore_network(invalid)

    def test_physics_routing_and_pair_configuration_validation(self):
        players,teams,ball=self.fixture(6)
        torch.manual_seed(12)
        architecture=dict(width=8,channels=17,dropout=0,instant_skip=True,geometry_attention=True)
        model=PaperPossession(**architecture).eval();pair=PhysicsPair()
        artifact=dict(kind='refined_contact',role_mode='none',input_mode='legacy17',rate=16,
            architecture=architecture,neural_state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()},
            neural_weight=.5,tree_model=ConstantTree(),boundary_model=ConstantOnset(),boundary_kind='state',
            pair_boundary_model=pair,pair_feature_mode='physics',pair_alpha=.4,penalty=16,boundary_scale=8)
        restored=restore_network(artifact,'cpu')
        result=predict_block(artifact,restored,players,teams,ball,[105,68])
        self.assertEqual(pair.input_shape,(len(ball),321))
        self.assertEqual(result.shape,(len(ball),))
        with self.assertRaises(ValueError):restore_network(dict(artifact,pair_feature_mode='unknown'))
        invalid=ConstantPair();invalid.classes_=np.arange(7)[::-1]
        with self.assertRaises(ValueError):restore_network(dict(artifact,pair_boundary_model=invalid))


if __name__=='__main__':unittest.main()
