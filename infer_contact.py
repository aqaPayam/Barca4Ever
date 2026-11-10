"""Local-only frozen contact-network/tree inference; no training or networking."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from possession_v2 import features
from learned_possession import LABELS
from infer_learned import load_tracking,tracking_blocks
from temporal_decoders import boundary_features,adaptive_decode
from temporal_state import decode
from contact_input import contact_features
from paper2023_model import PaperPossession,outputs,upsample,joint_probabilities
from torch_runtime import torch
from games import GAME,GAMES,require,source_frames
from barca4ever import write_csv
import joblib


def restore_network(artifact,device='cpu'):
    if artifact.get('kind')!='contact_fusion' or artifact.get('role_mode')!='none':raise ValueError('Unsupported frozen artifact')
    model=PaperPossession(**artifact['architecture'])
    model.load_state_dict({k:torch.from_numpy(v) for k,v in artifact['neural_state'].items()})
    return model.to(device).eval()


def predict_block(artifact,model,players,teams,ball,pitch,fps=25,device='cpu'):
    active=np.isfinite(players).all(2).any(0)
    pp=players[:,active];tt=teams[active]
    xx,samples=contact_features(pp,tt,ball,pitch,np.zeros(len(tt),bool),fps)
    assert not xx[:,:,6].any()
    marginal=upsample(outputs(model,xx,device),samples,len(ball))
    neural=joint_probabilities(marginal)
    x=features(players,teams,ball,pitch,fps)
    p=(1-artifact['neural_weight'])*artifact['tree_model'].predict_proba(x)+artifact['neural_weight']*neural
    bx=boundary_features(x)
    if artifact['boundary_kind']=='binary':
        b=artifact['boundary_model'].predict_proba(bx)[:,1]
        return adaptive_decode(p,b,artifact['penalty'],artifact['boundary_scale'])
    b=artifact['boundary_model'].predict_proba(bx)[:,1:]
    return decode(p,b,artifact['penalty'],artifact['boundary_scale'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--model',default='models/external_v4.joblib')
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
        pred=predict_block(artifact,model,players[start:end],teams,ball[start:end],pitch,fps,args.device)
        labels[start:end]=LABELS[pred]
    write_csv(args.out,labels)
    manifest=dict(team='Barca4Ever',rows=len(labels),model_kind='contact_fusion',device=args.device,
        model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
        training='External IDSSE only; no fitting, calibration or tuning during inference')
    Path(args.out+'.manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Frozen contact-model inference complete. Inputs, intermediate data and predictions remain local.')


if __name__=='__main__':main()
