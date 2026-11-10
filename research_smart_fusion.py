"""Match-separated external calibration of reliability gates versus V4.1."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import features
from paper2023_data import get,relative_features
from paper2023_model import outputs,upsample,joint_probabilities
from train_paper2023 import load_model
from infer_contact import restore_network
from temporal_decoders import boundary_features
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate
from smart_fusion import reliability,fit_gate,combine,head_weight
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'external_cache/smart_fusion'
MODELS=ROOT/'models/smart_fusion'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observations(match,h,noisy,seed,tree,boundary,model,device,fingerprint):
    CACHE.mkdir(parents=True,exist_ok=True)
    target=CACHE/f'{fingerprint[:16]}_{match}_{h}_{int(noisy)}_{seed}.npz'
    if target.exists():
        with np.load(target) as z:
            assert str(z['fingerprint'])==fingerprint
            return {k:z[k] for k in ['y','tree','neural','f','onset']}
    half=load(match)[h]
    players,ball=half['players'],half['ball']
    if noisy:players,ball=corrupt(players,ball,seed)
    x=features(players,half['teams'],ball,half['pitch'])
    objects=get(match,h,noisy,seed=seed,interpolation='short')
    assert np.array_equal(objects['full_y'],half['labels'])
    objects['x'][:,:,6]=0
    neural=joint_probabilities(upsample(outputs(model,relative_features(objects['x']),device),objects['samples'],len(half['labels'])))
    tree_p=tree.predict_proba(x)
    data=dict(y=half['labels'],tree=tree_p,neural=neural,f=reliability(tree_p,neural,x),
              onset=boundary.predict_proba(boundary_features(x))[:,1:])
    np.savez_compressed(target,**data,fingerprint=fingerprint)
    return data


def gate_for(kind,tag,calibration,fingerprint,base_matches):
    MODELS.mkdir(parents=True,exist_ok=True)
    target=MODELS/f'{kind}_{tag}.joblib'
    if target.exists():
        gate=joblib.load(target)
        assert gate['base_fingerprint']==fingerprint and gate['base_training_matches']==base_matches
        return gate
    gate=fit_gate(calibration,kind)
    gate.update(fitting_matches=['J03WR9'],base_training_matches=base_matches,base_fingerprint=fingerprint,
                protocol_sha256=digest(ROOT/'SMART_FUSION_PROTOCOL.md'),source_sha256=digest(ROOT/'smart_fusion.py'))
    joblib.dump(gate,target,compress=3)
    print('External gate fitted:',kind,tag,flush=True)
    return gate


def summarize(rows,name):
    subset=[r for r in rows if r['model']==name]
    return {str(noisy):aggregate([r for r in subset if r['noisy']==noisy]) for noisy in [False,True]}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['crosscheck','regression'],default='crosscheck')
    args=parser.parse_args()
    target=ROOT/f'external_reports/smart_fusion_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve comparison evidence')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    frozen_path=ROOT/'models/external_v4_1.joblib'
    frozen=joblib.load(frozen_path)
    assert frozen['boundary_scale']==8 and frozen['penalty']==16
    if args.stage=='crosscheck':
        candidates=['joint','factorized']
        folds=[(SPLITS['train'][:2],SPLITS['train'][2:],0),(SPLITS['train'][2:],SPLITS['train'][:2],1)]
        control_file='v4_decoder_crosscheck.json'
    else:
        cross=json.loads((ROOT/'external_reports/smart_fusion_crosscheck.json').read_text())
        assert cross['selected'] is not None,'Neither candidate passed CV; retain V4.1'
        candidates=[cross['selected']]
        folds=[(SPLITS['train'],SPLITS['test'],None)]
        control_file='v4_decoder_regression.json'
    controls=json.loads((ROOT/'external_reports'/control_file).read_text())['selected']['rows']
    lookup={(r['match'],r['half'],r['noisy']):r for r in controls}
    rows=[]
    for fit_matches,test_matches,fold in folds:
        assert not set(fit_matches+['J03WR9'])&set(test_matches)
        if fold is None:
            model=restore_network(frozen,device);tree=frozen['tree_model'];boundary=frozen['boundary_model']
            parts=[digest(frozen_path)];tag='full'
        else:
            path=ROOT/f'models/paper2023/rolefree_fold{fold+1}.pt'
            model,a=load_model(path,device)
            assert a['training_matches']==fit_matches and a['run_info']['role_mode']=='none'
            tp=ROOT/f'external_cache/research_fusion_cv/base_fold{fold+1}.joblib'
            bp=ROOT/f'external_cache/blended_boundaries/fold{fold+1}.joblib'
            tree=joblib.load(tp);boundary=joblib.load(bp)['state']
            parts=[digest(p) for p in [path,tp,bp]];tag=f'fold{fold+1}'
        fingerprint=hashlib.sha256(''.join(parts+[digest(ROOT/'smart_fusion.py')]).encode()).hexdigest()
        calibration=[]
        for h in [0,1]:
            for noisy in [False,True]:
                calibration.append(observations('J03WR9',h,noisy,8400+h,tree,boundary,model,device,fingerprint))
        gates={kind:gate_for(kind,tag,calibration,fingerprint,fit_matches) for kind in candidates}
        for m,match in enumerate(test_matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else 9500+h
                    d=observations(match,h,noisy,seed,tree,boundary,model,device,fingerprint)
                    probabilities={'v4.1':.5*(d['tree']+d['neural'])}
                    for kind,gate in gates.items():probabilities[kind]=combine(gate,d['tree'],d['neural'],d['f'])
                    for name,probability in probabilities.items():
                        assert np.isfinite(probability).all() and np.allclose(probability.sum(1),1,atol=1e-6)
                        result=score(d['y'],decode(probability,d['onset'],16,8))
                        if name=='v4.1':assert abs(result['spell_f1']-lookup[(match,h+1,noisy)]['spell_f1'])<1e-12,'V4.1 control failed'
                        rows.append(dict(match=match,half=h+1,noisy=noisy,model=name,**result))
                    print('External matched comparison:',tag,match,h+1,noisy,flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    summaries={name:summarize(rows,name) for name in ['v4.1']+candidates}
    accepted=[];gates={}
    for kind in candidates:
        checks={}
        for noisy in [False,True]:
            a,b=summaries['v4.1'][str(noisy)],summaries[kind][str(noisy)]
            checks[f'{noisy}_spell']=b['spell_f1']>a['spell_f1']
            checks[f'{noisy}_short']=b['short_hits']>=a['short_hits']
        gates[kind]=checks
        if all(checks.values()):accepted.append(kind)
    def rank(kind):
        return tuple(sum(summaries[kind][str(n)][key] for n in [False,True])/2 for key in ['spell_f1','macro_f1'])
    selected=max(accepted,key=rank) if accepted else None
    report=dict(status='Reused external development; WR9 fits gates, CV selects architecture; no competition access',
        baseline_sha256=digest(frozen_path),protocol_sha256=digest(ROOT/'SMART_FUSION_PROTOCOL.md'),
        source_sha256=digest(__file__),summary=summaries,gates=gates,selected=selected,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
