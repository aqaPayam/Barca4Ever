"""Controlled external folds for V2, TCN and their selected probability fusion."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from research_tcn import load_artifact,probabilities
from external_idsse import load,SPLITS
from learned_possession import corrupt,viterbi
from possession_v2 import features
from develop_v2 import fit_matches,score
import torch
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    torch.set_num_threads(4)
    target=ROOT/'external_reports/research_v3/fusion_crosscheck.json'
    if target.exists():raise RuntimeError('Preserve prior comparison')
    selected=json.loads((ROOT/'external_reports/research_v3/fusion.json').read_text())
    choice=max([t for t in selected['trials'] if t['kind']=='fusion'],key=lambda r:(r['spell_f1'],r['macro_f1']))
    steps=selected['checkpoint_step']
    folds=[(SPLITS['train'][:2],SPLITS['train'][2:]),(SPLITS['train'][2:],SPLITS['train'][:2])]
    rows=[]
    cache=ROOT/'external_cache/research_fusion_cv';cache.mkdir(exist_ok=True)
    for fold,(train,test) in enumerate(folds):
        print('Fusion cross-check fold '+str(fold+1),flush=True)
        base=fit_matches(train)
        joblib.dump(base,cache/('base_fold{}.joblib'.format(fold+1)))
        path=cache/('tcn_fold{}.pt'.format(fold+1))
        subprocess.run([sys.executable,str(ROOT/'research_tcn.py'),'--matches',*train,
                        '--steps',str(steps),'--fixed','--out',str(path)],check=True,cwd=ROOT)
        model,mean,std,checkpoint=load_artifact(path)
        for m,match in enumerate(test):
            for h,half in enumerate(load(match)):
                for noisy in [False,True]:
                    p,b=half['players'],half['ball']
                    if noisy:p,b=corrupt(p,b,11200+100*fold+10*m+h)
                    x=features(p,half['teams'],b,half['pitch'])
                    tree=base.predict_proba(x);neural,boundary=probabilities(model,x,mean,std)
                    fusion=(1-choice['tcn_weight'])*tree+choice['tcn_weight']*neural
                    for name,pred in [('v2',viterbi(tree,8)),('tcn',viterbi(neural,choice['penalty'])),
                                      ('fusion',viterbi(fusion,choice['penalty']))]:
                        r=dict(fold=fold+1,match=match,half=h+1,noisy=noisy,model=name,**score(half['labels'],pred))
                        rows.append(r)
                        print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
    target.write_text(json.dumps(dict(status='Controlled cross-validation on development data; no fresh blind test',
                       selection=choice,checkpoint_steps=steps,rows=rows),indent=2))
