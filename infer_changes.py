"""Local-only inference with a frozen externally trained state-change model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from possession_v2 import features
from learned_possession import LABELS
from infer_learned import load_tracking,tracking_blocks
from temporal_decoders import boundary_features
from temporal_state import decode
from games import GAME,GAMES,require,source_frames
from barca4ever import write_csv
import joblib


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--model',default='models/external_v4.joblib')
    parser.add_argument('--out',default='evaluation/results/Barca4Ever.csv')
    args=parser.parse_args()
    artifact=joblib.load(args.model)
    if artifact.get('kind') not in ['state_boundary','blended_boundary'] or artifact['feature_version']!=2:
        raise ValueError('Unsupported model artifact')
    meta,raw=require(['atd_meta','atd_raw'],data_dir=args.data_dir)
    pitch,fps=GAMES[GAME]['pitch'],GAMES[GAME]['fps']
    players,teams,ball=load_tracking(meta,raw,source_frames(),pitch)
    labels=np.full(len(ball),'X',dtype='<U1')
    for start,end in tracking_blocks(players,fps):
        x=features(players[start:end],teams,ball[start:end],pitch,fps)
        p=artifact['model'].predict_proba(x);bx=boundary_features(x)
        if artifact['kind']=='state_boundary':
            onset=artifact['boundary_model'].predict_proba(bx)[:,1:]
        else:
            binary=artifact['binary_model'].predict_proba(bx)[:,1,None]
            state=artifact['state_model'].predict_proba(bx)[:,1:]
            onset=(1-artifact['weight'])*binary+artifact['weight']*state
        labels[start:end]=LABELS[decode(p,onset,artifact['penalty'],artifact['boundary_scale'])]
    write_csv(args.out,labels)
    manifest=dict(team='Barca4Ever',rows=len(labels),feature_version=2,model_kind=artifact['kind'],
        model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
        training='IDSSE external matches only; no fitting or tuning during inference')
    Path(args.out+'.manifest.json').write_text(json.dumps(manifest,indent=2))
    print('State-change local inference complete. No restricted observations or predictions printed.')


if __name__=='__main__':main()
