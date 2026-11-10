"""Matched observability/rate experiments; no competition data access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from hz25_data import get
from hz25_model import load as load_network
from paper2023_model import outputs,upsample,joint_probabilities
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import features
from temporal_decoders import boundary_features
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(match,h,noisy,seed,rate,tree,boundary,model,device):
    half=load(match)[h];p,b=half['players'],half['ball']
    if noisy:p,b=corrupt(p,b,seed)
    x=features(p,half['teams'],b,half['pitch'])
    d=get(match,h,noisy,rate,seed)
    assert np.array_equal(d['full_y'],half['labels'])
    neural=joint_probabilities(upsample(outputs(model,d['x'],device),d['samples'],len(half['labels'])))
    return dict(y=half['labels'],neural=neural,tree=tree.predict_proba(x),onset=boundary.predict_proba(boundary_features(x))[:,1:])


def compare(d,weight):return score(d['y'],decode((1-weight)*d['tree']+weight*d['neural'],d['onset'],16,8))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','crosscheck','regression'],required=True)
    parser.add_argument('--rate',type=int,choices=[16,25])
    args=parser.parse_args()
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    base=joblib.load(ROOT/'models/external_v4_1.joblib')
    if args.stage=='validation':
        assert args.rate is not None
        target=ROOT/f'external_reports/hz25_validation_{args.rate}.json'
        if target.exists():raise RuntimeError('Preserve validation evidence')
        path=ROOT/f'models/hz25/rate{args.rate}.pt'
        model,a=load_network(path,device);assert a['training_matches']==SPLITS['train'] and a['rate']==args.rate
        trials={str(w):[] for w in [.25,.5,.75]}
        for h in [0,1]:
            for noisy in [False,True]:
                d=observe('J03WR9',h,noisy,8400+h,args.rate,base['tree_model'],base['boundary_model'],model,device)
                for key in trials:trials[key].append(dict(match='J03WR9',half=h+1,noisy=noisy,**compare(d,float(key))))
                print('External rate validation:',args.rate,h+1,noisy,flush=True)
        rows=[dict(rate=args.rate,weight=float(key),rows=value,**aggregate(value)) for key,value in trials.items()]
        target.write_text(json.dumps(dict(status='External reused validation selection',model_sha256=digest(path),trials=rows),indent=2)+'\n')
        print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in rows]),flush=True)
    else:
        selection=json.loads((ROOT/'external_reports/hz25_selection.json').read_text())
        assert selection['selected'] is not None,'No qualified validation candidate'
        choice=selection['selected'];rate=choice['rate'];weight=choice['weight']
        validation=json.loads((ROOT/f'external_reports/hz25_validation_{rate}.json').read_text())
        assert digest(ROOT/f'models/hz25/rate{rate}.pt')==validation['model_sha256']
        target=ROOT/f'external_reports/hz25_{args.stage}.json'
        if target.exists():raise RuntimeError('Preserve matched evidence')
        previous=json.loads((ROOT/('external_reports/v4_decoder_crosscheck.json' if args.stage=='crosscheck' else 'external_reports/v4_decoder_regression.json')).read_text())['selected']['rows']
        lookup={(r['match'],r['half'],r['noisy']):r for r in previous}
        folds=[(SPLITS['train'][:2],SPLITS['train'][2:],0),(SPLITS['train'][2:],SPLITS['train'][:2],1)] if args.stage=='crosscheck' else [(SPLITS['train'],SPLITS['test'],None)]
        rows=[]
        for train,test,fold in folds:
            tag='' if fold is None else f'_fold{fold+1}'
            path=ROOT/f'models/hz25/rate{rate}{tag}.pt'
            if not path.exists():
                parent=ROOT/f'models/paper2023/rolefree{tag}.pt'
                subprocess.run([sys.executable,str(ROOT/'hz25_train.py'),'--rate',str(rate),'--matches',*train,
                    '--initialize',str(parent),'--out',str(path)],check=True,cwd=ROOT)
            model,a=load_network(path,device);assert a['training_matches']==train and a['epochs']==2 and a['rate']==rate
            if fold is None:tree,boundary=base['tree_model'],base['boundary_model']
            else:
                tree=joblib.load(ROOT/f'external_cache/research_fusion_cv/base_fold{fold+1}.joblib')
                boundary=joblib.load(ROOT/f'external_cache/blended_boundaries/fold{fold+1}.joblib')['state']
            for m,match in enumerate(test):
                for h in [0,1]:
                    for noisy in [False,True]:
                        seed=11200+100*fold+10*m+h if fold is not None else 9500+h
                        d=observe(match,h,noisy,seed,rate,tree,boundary,model,device)
                        # Old neural path is separately cached from frozen V4.1; assert its control.
                        from research_v4_decoder import observations as old_observations
                        from train_paper2023 import load_model
                        from infer_contact import restore_network
                        oldpath=ROOT/f'models/paper2023/rolefree{tag}.pt'
                        oldmodel=(restore_network(base,device) if fold is None else load_model(oldpath,device)[0])
                        if fold is None:fingerprint=digest(ROOT/'models/external_v4.joblib')
                        else:
                            parts=[oldpath,ROOT/f'external_cache/research_fusion_cv/base_fold{fold+1}.joblib',ROOT/f'external_cache/blended_boundaries/fold{fold+1}.joblib']
                            fingerprint=hashlib.sha256(''.join(digest(p) for p in parts).encode()).hexdigest()
                        control=old_observations(match,h,noisy,seed,tree,boundary,oldmodel,device,fingerprint)
                        old=score(control['y'],decode(control['p'],control['b'],16,8))
                        ref=lookup[(match,h+1,noisy)]
                        assert abs(old['spell_f1']-ref['spell_f1'])<1e-12
                        rows.extend([dict(match=match,half=h+1,noisy=noisy,model='v4.1',**old),
                                     dict(match=match,half=h+1,noisy=noisy,model='candidate',**compare(d,weight))])
                        del oldmodel
                        print('External matched rate comparison:',match,h+1,noisy,flush=True)
            del model
            if device=='cuda':torch.cuda.empty_cache()
        summary={name:{str(n):aggregate([r for r in rows if r['model']==name and r['noisy']==n]) for n in [False,True]} for name in ['v4.1','candidate']}
        checks={}
        for n in [False,True]:
            a,b=summary['v4.1'][str(n)],summary['candidate'][str(n)]
            checks[f'{n}_spell']=b['spell_f1']>a['spell_f1'];checks[f'{n}_short']=b['short_hits']>=a['short_hits']
        target.write_text(json.dumps(dict(status='External development comparison, not fresh blind test',choice=choice,
            summary=summary,gates=checks,accepted=all(checks.values()),rows=rows),indent=2)+'\n')
        print(json.dumps(dict(summary=summary,gates=checks)),flush=True)
