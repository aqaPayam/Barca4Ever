"""Local-only inference using a frozen externally trained boundary-aware model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from possession_v2 import features
from learned_possession import LABELS
from infer_learned import load_tracking,tracking_blocks
from barca4ever import write_csv
from temporal_decoders import boundary_features,adaptive_decode
from games import GAME,GAMES,require,source_frames
import joblib


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--model',default='models/external_v3.joblib')
    parser.add_argument('--out',default='evaluation/results/Barca4Ever.csv')
    args=parser.parse_args()
    artifact=joblib.load(args.model)
    if artifact.get('kind')!='adaptive_boundary' or artifact['feature_version']!=2:
        raise ValueError('Unsupported model artifact')
    meta,raw=require(['atd_meta','atd_raw'],data_dir=args.data_dir)
    pitch,fps=GAMES[GAME]['pitch'],GAMES[GAME]['fps']
    players,teams,ball=load_tracking(meta,raw,source_frames(),pitch)
    labels=np.full(len(ball),'X',dtype='<U1')
    for start,end in tracking_blocks(players,fps):
        x=features(players[start:end],teams,ball[start:end],pitch,fps)
        p=artifact['model'].predict_proba(x)
        b=artifact['boundary_model'].predict_proba(boundary_features(x))[:,1]
        labels[start:end]=LABELS[adaptive_decode(p,b,artifact['penalty'],artifact['boundary_scale'])]
    write_csv(args.out,labels)
    manifest=dict(team='Barca4Ever',rows=len(labels),feature_version=2,model_kind='adaptive_boundary',
                  model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
                  training='External IDSSE matches only; no fitting or tuning during inference')
    Path(args.out+'.manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Boundary-aware local inference complete. No restricted observations or predictions printed.')


if __name__=='__main__':main()
