"""Freeze and compare full paper-based models on reused external regression games."""
import argparse
import json
from pathlib import Path
import numpy as np
from train_paper2023 import load_model
from paper2023_data import get,relative_features
from paper2023_model import outputs,upsample
from crosscheck_paper2023 import prediction
from external_idsse import SPLITS
from develop_v2 import score
from torch_runtime import torch

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe',choices=['paper','adapted','relative','rolefree'],required=True)
    args=parser.parse_args()
    target=ROOT/f'external_reports/paper2023_regression_{args.recipe}.json'
    if target.exists():raise RuntimeError('Preserve prior external evaluation evidence')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    val=json.loads((ROOT/f'models/paper2023/{args.recipe}.validation.json').read_text())
    model,artifact=load_model(ROOT/f'models/paper2023/{args.recipe}.pt',device)
    old=json.loads((ROOT/'external_reports/research_v3/boundary_regression.json').read_text())['rows']
    rows=[dict(r,model='v3') for r in old if r['model']=='learned_boundary']
    for match in SPLITS['test']:
        for h in [0,1]:
            for noisy in [False,True]:
                d=get(match,h,noisy,seed=9500+h,interpolation=artifact['interpolation'])
                if args.recipe=='rolefree':d['x'][:,:,6]=0
                if args.recipe in ['relative','rolefree']:d['x']=relative_features(d['x'])
                marginal=upsample(outputs(model,d['x'],device),d['samples'],len(d['full_y']))
                methods=[(args.recipe,val['selected']['penalty'])]
                if args.recipe=='paper':methods.append(('paper_raw','hierarchical'))
                for name,penalty in methods:
                    p=prediction(marginal,penalty);y=d['full_y'];valid=y<3
                    r=dict(match=match,half=h+1,noisy=noisy,model=name,
                           global_accuracy=float((p[valid]==y[valid]).mean()),**score(y,p))
                    rows.append(r);print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
    target.write_text(json.dumps(dict(status='Frozen models on previously inspected external regression games, not fresh blind testing',
                        selected_epoch=artifact['epoch'],selected_penalty=val['selected']['penalty'],
                        model_sha256=val['model_sha256'],rows=rows),indent=2))
