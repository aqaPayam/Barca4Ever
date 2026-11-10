"""Small external-validation fusion comparison; no competition data."""
import json
from pathlib import Path
import numpy as np
from research_tcn import load_artifact,probabilities
from research_data import validation
from learned_possession import viterbi
from research_trees import adaptive_decode
from develop_v2 import score
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    model,mean,std,checkpoint=load_artifact(ROOT/'models/research_v3/tcn.pt')
    tree=joblib.load(ROOT/'models/external_v2.joblib')
    val=validation();outputs=[]
    for h,noisy,d in val:
        neural,boundary=probabilities(model,d['x'],mean,std)
        outputs.append((tree['model'].predict_proba(d['x']),neural,boundary))
    trials=[]
    configs=[('fusion',weight,penalty,0) for weight in [.25,.5] for penalty in [4,8]]
    # Explicit exploratory extension: assess the already-trained boundary head.
    configs += [('tcn_boundary',1.,p,4) for p in [8,16]]
    for kind,weight,penalty,scale in configs:
        rows=[]
        for (h,noisy,d),(tree_p,neural,boundary) in zip(val,outputs):
            p=(1-weight)*tree_p+weight*neural
            pred=viterbi(p,penalty) if not scale else adaptive_decode(p,boundary,penalty,scale)
            rows.append(dict(half=h+1,noisy=noisy,**score(d['y'],pred)))
        r=dict(kind=kind,tcn_weight=weight,penalty=penalty,boundary_scale=scale,
               spell_f1=float(np.mean([a['spell_f1'] for a in rows])),
               macro_f1=float(np.mean([a['macro_f1'] for a in rows])),rows=rows)
        trials.append(r)
        print(json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
    best=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1']))
    (ROOT/'external_reports/research_v3/fusion.json').write_text(json.dumps(dict(
        status='Validation selection with exploratory boundary-head decoding; not a blind test',
        checkpoint_step=checkpoint['step'],trials=trials,selected=best),indent=2))
