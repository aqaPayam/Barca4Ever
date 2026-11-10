"""Small externally selected fusion: contact network + frozen tree estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from train_paper2023 import load_model
from paper2023_data import get,relative_features
from paper2023_model import outputs,upsample,joint_probabilities
from possession_v2 import features
from temporal_decoders import boundary_features,adaptive_decode
from research_state_boundaries import decode
from external_idsse import load,SPLITS
from learned_possession import corrupt
from develop_v2 import score
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


def predict(prob,boundary,kind):
    return adaptive_decode(prob,boundary[:,0],16,4) if kind=='binary' else decode(prob,boundary,16,4)


def evaluate_inputs(matches,model,device,tree,binary,state,validation=False,zero_keeper=False):
    results=[]
    for match in matches:
        for h,half in enumerate(load(match)):
            for noisy in [False,True]:
                seed=8400+h if validation else 9500+h
                p,b=half['players'],half['ball']
                if noisy:p,b=corrupt(p,b,seed)
                x=features(p,half['teams'],b,half['pitch']);bx=boundary_features(x)
                d=get(match,h,noisy,seed=seed,interpolation='short')
                if zero_keeper:d['x'][:,:,6]=0
                marginal=upsample(outputs(model,relative_features(d['x']),device),d['samples'],len(d['full_y']))
                assert np.array_equal(d['full_y'],half['labels'])
                results.append(dict(match=match,half=h+1,noisy=noisy,y=half['labels'],
                    tree=tree.predict_proba(x),neural=joint_probabilities(marginal),
                    binary=binary.predict_proba(bx)[:,1:2],state=state.predict_proba(bx)[:,1:]))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['validation','regression'],default='validation')
    parser.add_argument('--recipe',choices=['relative','rolefree'],default='relative')
    args=parser.parse_args()
    name='fusion' if args.recipe=='relative' else 'fusion_rolefree'
    target=ROOT/(f'models/paper2023/{name}.validation.json' if args.stage=='validation' else
                 f'external_reports/paper2023_regression_{name}.json')
    if target.exists():raise RuntimeError('Preserve previous fusion evidence')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    neural_path=ROOT/f'models/paper2023/{args.recipe}.pt'
    model,artifact=load_model(neural_path,device)
    v3=joblib.load(ROOT/'models/external_v3.joblib');state=joblib.load(ROOT/'models/state_boundary.joblib')
    matches=SPLITS['validation'] if args.stage=='validation' else SPLITS['test']
    results=evaluate_inputs(matches,model,device,v3['model'],v3['boundary_model'],state['boundary_model'],args.stage=='validation',args.recipe=='rolefree')
    if args.stage=='validation':configs=[dict(neural_weight=w,boundary_kind=k) for w in [.25,.5,1.] for k in ['binary','state']]
    else:configs=[json.loads((ROOT/f'models/paper2023/{name}.validation.json').read_text())['selected']]
    trials=[]
    for config in configs:
        rows=[];weight=config['neural_weight'];kind=config['boundary_kind']
        for d in results:
            prob=(1-weight)*d['tree']+weight*d['neural']
            p=predict(prob,d[kind],kind)
            rows.append(dict(match=d['match'],half=d['half'],noisy=d['noisy'],model='fusion',**score(d['y'],p)))
        trial=dict(neural_weight=weight,boundary_kind=kind,rows=rows,
            spell_f1=float(np.mean([r['spell_f1'] for r in rows])),macro_f1=float(np.mean([r['macro_f1'] for r in rows])))
        trials.append(trial);print(json.dumps({k:v for k,v in trial.items() if k!='rows'}),flush=True)
    selected=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1']))
    target.write_text(json.dumps(dict(status='External validation selection' if args.stage=='validation' else 'Reused external regression, not a fresh blind test',
        trials=trials,selected=selected,rows=selected['rows'],
        neural_sha256=hashlib.sha256(neural_path.read_bytes()).hexdigest(),recipe=args.recipe),indent=2))
