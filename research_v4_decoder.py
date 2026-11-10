"""Externally selected decoder-only trial against frozen V4; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from external_idsse import load
from learned_possession import corrupt
from possession_v2 import features
from paper2023_data import get,relative_features
from paper2023_model import outputs,upsample,joint_probabilities
from infer_contact import restore_network
from train_paper2023 import load_model
from temporal_decoders import boundary_features
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'external_cache/v4_decoder'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observations(match,h,noisy,seed,tree,boundary,model,device,fingerprint):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/f'{fingerprint[:16]}_{match}_{h}_{int(noisy)}_{seed}.npz'
    if path.exists():
        with np.load(path) as data:
            assert str(data['fingerprint'])==fingerprint
            return {k:data[k] for k in ['y','p','b']}
    half=load(match)[h]
    players,ball=half['players'],half['ball']
    if noisy:players,ball=corrupt(players,ball,seed)
    x=features(players,half['teams'],ball,half['pitch'])
    objects=get(match,h,noisy,seed=seed,interpolation='short')
    objects['x'][:,:,6]=0
    assert np.array_equal(objects['full_y'],half['labels'])
    neural=joint_probabilities(upsample(outputs(model,relative_features(objects['x']),device),objects['samples'],len(half['labels'])))
    data=dict(y=half['labels'],p=.5*(tree.predict_proba(x)+neural),b=boundary.predict_proba(boundary_features(x))[:,1:])
    np.savez_compressed(path,**data,fingerprint=fingerprint)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['validation','crosscheck','regression'],default='validation')
    args=parser.parse_args()
    target=ROOT/f'external_reports/v4_decoder_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve trial evidence')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    frozen_path=ROOT/'models/external_v4.joblib'
    frozen=joblib.load(frozen_path)
    assert frozen['neural_weight']==.5 and frozen['boundary_kind']=='state'
    if args.stage=='validation':
        configs=[dict(base=base,scale=scale) for base in [8,12,16] for scale in [4,8]]
        controls=json.loads((ROOT/'models/paper2023/fusion_rolefree.validation.json').read_text())['selected']['rows']
    else:
        selection=json.loads((ROOT/'external_reports/v4_decoder_validation.json').read_text())
        assert selection['selected']['spell_f1']>selection['baseline']['spell_f1'],'No validation improvement'
        chosen={k:selection['selected'][k] for k in ['base','scale']}
        configs=[dict(base=16,scale=4)]
        if chosen not in configs:configs.append(chosen)
        file='paper2023_rolefree_crosscheck.json' if args.stage=='crosscheck' else 'paper2023_regression_fusion_rolefree.json'
        controls=[r for r in json.loads((ROOT/'external_reports'/file).read_text())['rows'] if r['model']=='fusion']
    lookup={(r.get('match','J03WR9'),r['half'],r['noisy']):r for r in controls}
    all_rows=[[] for _ in configs]
    folds=[(['J03WPY','J03WQQ'],0),(['J03WOH','J03WOY'],1)] if args.stage=='crosscheck' else [(['J03WR9'] if args.stage=='validation' else ['J03WMX','J03WN1'],None)]
    for matches,fold in folds:
        if fold is None:
            model=restore_network(frozen,device)
            tree,boundary=frozen['tree_model'],frozen['boundary_model']
            fingerprint=digest(frozen_path)
        else:
            path=ROOT/f'models/paper2023/rolefree_fold{fold+1}.pt'
            model,artifact=load_model(path,device)
            expected=['J03WOH','J03WOY'] if fold==0 else ['J03WPY','J03WQQ']
            assert artifact['training_matches']==expected and artifact['run_info']['role_mode']=='none'
            tp=ROOT/f'external_cache/research_fusion_cv/base_fold{fold+1}.joblib'
            bp=ROOT/f'external_cache/blended_boundaries/fold{fold+1}.joblib'
            tree=joblib.load(tp);boundary=joblib.load(bp)['state']
            fingerprint=hashlib.sha256(''.join(digest(p) for p in [path,tp,bp]).encode()).hexdigest()
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else (8400 if args.stage=='validation' else 9500)+h
                    data=observations(match,h,noisy,seed,tree,boundary,model,device,fingerprint)
                    control=None
                    for i,config in enumerate(configs):
                        prediction=decode(data['p'],data['b'],config['base'],config['scale'])
                        result=score(data['y'],prediction)
                        if config==dict(base=16,scale=4):
                            control=result
                            assert abs(result['spell_f1']-lookup[(match,h+1,noisy)]['spell_f1'])<1e-12,'V4 control failed'
                        all_rows[i].append(dict(match=match,half=h+1,noisy=noisy,**result))
                    assert control is not None
                    print('External decoder comparison complete:',match,h+1,noisy,flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    trials=[dict(**config,rows=rows,**aggregate(rows)) for config,rows in zip(configs,all_rows)]
    baseline=next(t for t in trials if t['base']==16 and t['scale']==4)
    selected=max(trials,key=lambda t:(t['spell_f1'],t['macro_f1'])) if args.stage=='validation' else next(t for t in trials if (t['base'],t['scale'])==(chosen['base'],chosen['scale']))
    report=dict(status='External development decoder-only trial; no refitting or competition access',
        frozen_model_sha256=digest(frozen_path),protocol_sha256=digest(ROOT/'V4_DECODER_PROTOCOL.md'),
        source_sha256=digest(__file__),baseline=baseline,selected=selected,trials=trials)
    target.write_text(json.dumps(report,indent=2)+'\n')
    for name,trial in [('baseline',baseline),('selected',selected)]:
        print(name,json.dumps({k:v for k,v in trial.items() if k!='rows'}),flush=True)
