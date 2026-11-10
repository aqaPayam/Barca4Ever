"""Local-only inference for a qualified contact/boundary refinement.

Importing this module reads no competition inputs. Only explicit CLI execution
loads tracking. No training, calibration, networking or quality scoring occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from possession_v2 import features
from learned_possession import LABELS
from infer_learned import load_tracking,tracking_blocks
from contact_input import contact_features
from hz25_data import object_features as confidence_object_features
from paper2023_model import PaperPossession,outputs,upsample,joint_probabilities
from temporal_decoders import boundary_features
from temporal_state import decode as decode_state
import boundary_refinement
from torch_runtime import torch
from games import GAME,GAMES,require,source_frames
from barca4ever import write_csv
import joblib


def confidence_features(players,teams,ball,pitch,rate=25,fps=25):
    """Retain every actor, in ball/ref, home-group, away-group order.

    Each source transform supplies20 channels. Relative geometry and global
    observability are identical for an actor regardless of its padded group.
    """
    players=np.asarray(players);teams=np.asarray(teams);ball=np.asarray(ball)
    if players.ndim!=3 or players.shape[2]!=2 or ball.shape!=(len(players),2):
        raise ValueError('Invalid tracking dimensions')
    if teams.shape!=(players.shape[1],) or not np.isin(teams,[0,1]).all():
        raise ValueError('Invalid team mapping')
    if rate not in [16,25]:raise ValueError('Unsupported neural sample rate')
    ids=[np.flatnonzero(teams==team) for team in [0,1]]
    groups=max(1,(max(map(len,ids))+16)//17)
    blocks=[];samples=None
    for group in range(groups):
        take=np.r_[ids[0][group*17:(group+1)*17],ids[1][group*17:(group+1)*17]]
        block,current=confidence_object_features(players[:,take],teams[take],ball,pitch,
                                                 target_fps=rate,source_fps=fps)
        if samples is None:samples=current
        elif not np.array_equal(samples,current):raise RuntimeError('Inconsistent grouped time alignment')
        blocks.append(block)
    x=np.concatenate([blocks[0][:,:2]]+[block[:,2:19] for block in blocks]+
                     [block[:,19:36] for block in blocks],axis=1)
    if x.shape[2]!=20 or x[:,:,6].any():raise RuntimeError('Invalid confidence input contract')
    return x,samples


def _spec(artifact):
    if artifact.get('kind')!='refined_contact' or artifact.get('role_mode')!='none':
        raise ValueError('Unsupported refined artifact')
    mode=artifact.get('input_mode');rate=artifact.get('rate')
    if mode not in ['legacy17','confidence20'] or rate not in [16,25]:
        raise ValueError('Missing or unsupported input mode/rate')
    if mode=='legacy17' and rate!=16:
        raise ValueError('Legacy adapter requires its original16Hz sampling')
    expected=17 if mode=='legacy17' else 20
    if artifact['architecture']['channels']!=expected:
        raise ValueError('Neural architecture and input mode disagree')
    if not artifact['architecture'].get('geometry_attention'):
        raise ValueError('Grouped inputs require missing-object attention masking')
    if artifact.get('boundary_kind','state')!='state':
        raise ValueError('Refined inference requires state boundary scores')
    if artifact.get('penalty',16)!=16 or artifact.get('boundary_scale',8)!=8:
        raise ValueError('Unsupported refinement decoding configuration')
    if not 0<=artifact['neural_weight']<=1:raise ValueError('Invalid neural mixture weight')
    if artifact.get('pair_feature_mode','signed') not in ['signed','physics']:
        raise ValueError('Unsupported pair-boundary feature mode')
    if np.asarray(artifact['tree_model'].classes_).tolist()!=[0,1,2]:
        raise ValueError('Unexpected possession-class order')
    if np.asarray(artifact['boundary_model'].classes_).tolist()!=[0,1,2,3]:
        raise ValueError('Unexpected state-boundary class order')
    if artifact.get('pair_boundary_model') is not None:
        if 'pair_alpha' not in artifact or not 0<=artifact['pair_alpha']<=1:
            raise ValueError('Missing or invalid pair-boundary mixture weight')
        if np.asarray(artifact['pair_boundary_model'].classes_).tolist()!=list(range(7)):
            raise ValueError('Unexpected pair-boundary class order')
    elif artifact.get('pair_alpha',0)!=0:
        raise ValueError('Pair mixture weight requires a pair-boundary model')
    return mode,rate


def restore_network(artifact,device='cpu'):
    _spec(artifact)
    if not all(isinstance(value,np.ndarray) for value in artifact['neural_state'].values()):
        raise ValueError('Expected serialized NumPy neural state')
    model=PaperPossession(**artifact['architecture'])
    model.load_state_dict({key:torch.from_numpy(value) for key,value in artifact['neural_state'].items()})
    return model.to(device).eval()


def predict_block(artifact,model,players,teams,ball,pitch,fps=25,device='cpu'):
    mode,rate=_spec(artifact)
    players=np.asarray(players);teams=np.asarray(teams);ball=np.asarray(ball)
    if not len(ball):return np.empty(0,np.int8)
    active=np.isfinite(players).all(2).any(0)
    p=players[:,active];t=np.asarray(teams)[active]
    if mode=='legacy17':
        xx,samples=contact_features(p,t,ball,pitch,np.zeros(len(t),bool),fps)
    else:
        xx,samples=confidence_features(p,t,ball,pitch,rate,fps)
    if xx[:,:,6].any():raise RuntimeError('Keeper roles must remain zero')
    neural=joint_probabilities(upsample(outputs(model,xx,device),samples,len(ball)))
    x=features(players,teams,ball,pitch,fps)
    probability=(1-artifact['neural_weight'])*artifact['tree_model'].predict_proba(x)+artifact['neural_weight']*neural
    destination=artifact['boundary_model'].predict_proba(boundary_features(x))[:,1:]
    if artifact.get('pair_boundary_model') is None:
        return decode_state(probability,destination,16,8)
    if artifact.get('pair_feature_mode','signed')=='physics':
        import boundary_physics
        physical=boundary_physics.physics_features(players,teams,ball,pitch,fps)
        pair_input=boundary_physics.combined_features(x,physical)
        pair=boundary_physics.matrix_scores(artifact['pair_boundary_model'],pair_input)
    else:
        pair=boundary_refinement.matrix_scores(artifact['pair_boundary_model'],x)
    blended=boundary_refinement.blend_scores(destination,pair,artifact['pair_alpha'])
    return boundary_refinement.decode(probability,blended,16,8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--model',default='models/external_v5.joblib')
    parser.add_argument('--out',default='evaluation/results/Barca4Ever.csv')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(4)
    artifact=joblib.load(args.model)
    model=restore_network(artifact,args.device)
    meta,raw=require(['atd_meta','atd_raw'],data_dir=args.data_dir)
    pitch,fps=GAMES[GAME]['pitch'],GAMES[GAME]['fps']
    players,teams,ball=load_tracking(meta,raw,source_frames(),pitch)
    labels=np.full(len(ball),'X',dtype='<U1')
    for start,end in tracking_blocks(players,fps):
        prediction=predict_block(artifact,model,players[start:end],teams,ball[start:end],pitch,fps,args.device)
        labels[start:end]=LABELS[prediction]
    write_csv(args.out,labels)
    manifest=dict(team='Barca4Ever',rows=len(labels),model_kind='refined_contact',device=args.device,
        input_mode=artifact['input_mode'],rate=artifact['rate'],
        model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
        inference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training='External IDSSE only; no fitting, calibration or tuning during inference')
    Path(args.out+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen refined inference complete. Inputs, intermediates and predictions remain local.')


if __name__=='__main__':main()
