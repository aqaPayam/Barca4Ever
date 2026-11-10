"""Fit contact network within external folds; compare it and frozen fusion to V3."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from train_paper2023 import load_model
from paper2023_data import get,relative_features
from paper2023_model import outputs,upsample,joint_probabilities
from crosscheck_paper2023 import prediction
from fuse_paper2023 import predict
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import features
from temporal_decoders import boundary_features,adaptive_decode
from develop_v2 import score
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rolefree',action='store_true')
    args=parser.parse_args()
    recipe='rolefree' if args.rolefree else 'relative'
    fusion_name='fusion_rolefree' if args.rolefree else 'fusion'
    target=ROOT/('external_reports/paper2023_rolefree_crosscheck.json' if args.rolefree else 'external_reports/paper2023_crosscheck.json')
    if target.exists():raise RuntimeError('Preserve prior cross-check evidence')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    selection=json.loads((ROOT/f'models/paper2023/{recipe}.validation.json').read_text())
    fusion=json.loads((ROOT/f'models/paper2023/{fusion_name}.validation.json').read_text())['selected']
    full=torch.load(ROOT/f'models/paper2023/{recipe}.pt',map_location='cpu',weights_only=True)
    epochs,batch=selection['selected_epoch'],full['run_info']['batch_size']
    previous=json.loads((ROOT/'external_reports/research_v3/boundary_crosscheck.json').read_text())['rows']
    controls={(r['match'],r['half'],r['noisy']):r for r in previous if r['model']=='learned_boundary'}
    rows=[]
    folds=[(SPLITS['train'][:2],SPLITS['train'][2:]),(SPLITS['train'][2:],SPLITS['train'][:2])]
    for fold,(train,test) in enumerate(folds):
        path=ROOT/f'models/paper2023/{recipe}_fold{fold+1}.pt'
        if not path.exists():
            initialize=[]
            if args.rolefree:
                parent=ROOT/f'models/paper2023/relative_fold{fold+1}.pt'
                parent_data=torch.load(parent,map_location='cpu',weights_only=True)
                assert parent_data['training_matches']==train and parent_data['epoch']==8
                initialize=['--initialize',str(parent)]
            subprocess.run([sys.executable,str(ROOT/'train_paper2023.py'),'--recipe',recipe,
                '--matches',*train,'--fixed','--epochs',str(epochs),'--batch',str(batch),
                '--width',str(full['architecture']['width']),'--seed',str(full['seed']),
                *initialize,'--out',str(path)],check=True,cwd=ROOT)
        model,artifact=load_model(path,device)
        assert artifact['training_matches']==train and artifact['epoch']==epochs,'Incomplete or mismatched fold artifact'
        assert artifact['run_info']['batch_size']==batch
        base_path=ROOT/f'external_cache/research_fusion_cv/base_fold{fold+1}.joblib'
        boundary_path=ROOT/f'external_cache/blended_boundaries/fold{fold+1}.joblib'
        if not base_path.exists():
            from develop_v2 import fit_matches
            base_path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(fit_matches(train),base_path)
        if not boundary_path.exists():
            from research_trees import fit_boundary
            from research_state_boundaries import fit as fit_state
            boundary_path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(dict(binary=fit_boundary(train),state=fit_state(train)),boundary_path)
        base=joblib.load(base_path)
        boundary=joblib.load(boundary_path)
        for m,match in enumerate(test):
            for h,half in enumerate(load(match)):
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h
                    p,b=half['players'],half['ball']
                    if noisy:p,b=corrupt(p,b,seed)
                    x=features(p,half['teams'],b,half['pitch']);bx=boundary_features(x)
                    tree=base.predict_proba(x)
                    binary=boundary['binary'].predict_proba(bx)[:,1:2]
                    state=boundary['state'].predict_proba(bx)[:,1:]
                    d=get(match,h,noisy,seed=seed,interpolation='short')
                    if args.rolefree:d['x'][:,:,6]=0
                    assert np.array_equal(d['full_y'],half['labels'])
                    marginal=upsample(outputs(model,relative_features(d['x']),device),d['samples'],len(d['full_y']))
                    neural=joint_probabilities(marginal)
                    mixed=(1-fusion['neural_weight'])*tree+fusion['neural_weight']*neural
                    pred_v3=adaptive_decode(tree,binary[:,0],16,4)
                    v3_score=score(half['labels'],pred_v3)
                    assert abs(v3_score['spell_f1']-controls[(match,h+1,noisy)]['spell_f1'])<1e-12,'V3 control failed exact reproduction'
                    candidates=[('v3',pred_v3),(recipe,prediction(marginal,selection['selected']['penalty'])),
                                ('fusion',predict(mixed,binary if fusion['boundary_kind']=='binary' else state,fusion['boundary_kind']))]
                    for name,pred in candidates:
                        r=dict(fold=fold+1,match=match,half=h+1,noisy=noisy,model=name,**score(half['labels'],pred))
                        rows.append(r);print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    target.write_text(json.dumps(dict(status='External development cross-validation; not fresh blind testing',
        neural_epochs=epochs,neural_batch=batch,fusion=fusion,rows=rows),indent=2))
