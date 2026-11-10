"""Frozen symmetry averaging, external matched comparisons, CPU-only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from external_idsse import SPLITS
from paper2023_data import relative_features
from paper2023_model import PaperPossession,outputs,upsample,joint_probabilities
from tta_features import team_swap,reflect_pitch,restore_team_probabilities,average_predictions
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_network(tag):
    path=ROOT/f'models/paper2023/rolefree{tag}.pt'
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    assert artifact['architecture']['channels']==17 and artifact['run_info']['role_mode']=='none'
    model=PaperPossession(**artifact['architecture']).cpu().eval()
    model.load_state_dict(artifact['state_dict'])
    if not tag:
        fingerprint=hashlib.sha256((digest(ROOT/'models/external_v4_1.joblib')+digest(ROOT/'smart_fusion.py')).encode()).hexdigest()
    else:
        parent=joblib.load(ROOT/f'models/smart_fusion/joint{tag}.joblib')
        assert parent['base_training_matches']==artifact['training_matches']
        fingerprint=parent['base_fingerprint']
    return model,artifact,digest(path),fingerprint


def observations(match,h,noisy,seed,model,model_hash,component_fingerprint,candidates):
    raw_path=ROOT/f'external_cache/paper2023/{match}_{h}_{int(noisy)}_{seed}_short_v1.npz'
    component_path=ROOT/f'external_cache/smart_fusion/{component_fingerprint[:16]}_{match}_{h}_{int(noisy)}_{seed}.npz'
    with np.load(raw_path) as source:
        raw=source['x'].copy();samples=source['samples'].copy()
    raw[:,:,6]=0
    with np.load(component_path) as source:
        assert str(source['fingerprint'])==component_fingerprint
        data={k:source[k] for k in ['y','tree','neural','onset']}
    transforms=['original']
    if any(c in ['swap','both'] for c in candidates):transforms.append('swap')
    if any(c in ['reflection','both'] for c in candidates):transforms.append('reflection')
    folder=ROOT/'external_cache/tta';folder.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(''.join([model_hash,digest(raw_path),digest(ROOT/'tta_features.py'),digest(ROOT/'paper2023_data.py'),digest(ROOT/'paper2023_model.py')]).encode()).hexdigest()
    predictions={}
    for transform in transforms:
        target=folder/f'{fingerprint[:20]}_{transform}.npz'
        if target.exists():
            with np.load(target) as cache:
                assert str(cache['fingerprint'])==fingerprint
                predictions[transform]=cache['prediction']
            continue
        xx=raw if transform=='original' else team_swap(raw) if transform=='swap' else reflect_pitch(raw)
        marginal=upsample(outputs(model,relative_features(xx),'cpu'),samples,len(data['y']))
        prediction=joint_probabilities(marginal)
        if transform=='swap':prediction=restore_team_probabilities(prediction)
        assert np.isfinite(prediction).all() and np.allclose(prediction.sum(1),1,atol=1e-6)
        np.savez_compressed(target,fingerprint=fingerprint,prediction=prediction)
        predictions[transform]=prediction
        print('CPU external transform computed:',match,h+1,noisy,transform,flush=True)
    data['predictions']=predictions
    return data


def scored(data,neural):
    return score(data['y'],decode(.5*(data['tree']+neural),data['onset'],16,8))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','crosscheck','regression'],required=True)
    args=parser.parse_args();target=ROOT/f'external_reports/tta_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve TTA trial evidence')
    torch.set_num_threads(2)
    if args.stage=='validation':
        candidates=['swap','reflection','both'];folds=[(SPLITS['train'],['J03WR9'],None)]
        reference=json.loads((ROOT/'external_reports/v4_decoder_validation.json').read_text())['selected']['rows']
    else:
        selection=json.loads((ROOT/'external_reports/tta_validation.json').read_text())
        assert selection['selected'],'No TTA candidate passed validation'
        candidates=[selection['selected']]
        if args.stage=='regression':assert json.loads((ROOT/'external_reports/tta_crosscheck.json').read_text())['accepted'],'TTA CV failed'
        folds=[(SPLITS['train'][:2],SPLITS['train'][2:],0),(SPLITS['train'][2:],SPLITS['train'][:2],1)] if args.stage=='crosscheck' else [(SPLITS['train'],SPLITS['test'],None)]
        reference=json.loads((ROOT/('external_reports/v4_decoder_crosscheck.json' if args.stage=='crosscheck' else 'external_reports/v4_decoder_regression.json')).read_text())['selected']['rows']
    lookup={(r.get('match','J03WR9'),r['half'],r['noisy']):r for r in reference}
    rows=[];model_hashes={}
    for fit_matches,test_matches,fold in folds:
        assert not set(fit_matches)&set(test_matches)
        tag='' if fold is None else f'_fold{fold+1}'
        model,artifact,model_hash,fingerprint=load_network(tag)
        assert artifact['training_matches']==fit_matches
        model_hashes[tag or 'full']=model_hash
        for m,match in enumerate(test_matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else (8400 if args.stage=='validation' else 9500)+h
                    data=observations(match,h,noisy,seed,model,model_hash,fingerprint,candidates)
                    baseline=scored(data,data['predictions']['original'])
                    ref=lookup[(match,h+1,noisy)]
                    assert baseline['spell_f1']==ref['spell_f1'],'CPU V4.1 spell-score reproduction failed'
                    assert baseline['macro_f1']==ref['macro_f1'],'CPU V4.1 macro-score reproduction failed'
                    assert baseline['recall_by_duration']==ref['recall_by_duration'],'CPU V4.1 short-recall reproduction failed'
                    rows.append(dict(match=match,half=h+1,noisy=noisy,model='v4.1',**baseline))
                    for candidate in candidates:
                        rows.append(dict(match=match,half=h+1,noisy=noisy,model=candidate,**scored(data,average_predictions(data['predictions'],candidate))))
                    print('CPU external TTA case scored:',match,h+1,noisy,flush=True)
        del model
    summary={name:{str(noisy):aggregate([r for r in rows if r['model']==name and r['noisy']==noisy]) for noisy in [False,True]} for name in ['v4.1']+candidates}
    gates={};eligible=[]
    for candidate in candidates:
        if args.stage=='validation':
            old=aggregate([r for r in rows if r['model']=='v4.1']);new=aggregate([r for r in rows if r['model']==candidate])
            checks=dict(spell=new['spell_f1']>old['spell_f1'],short=new['short_hits']>=old['short_hits'])
        else:
            checks={}
            for noisy in [False,True]:
                a,b=summary['v4.1'][str(noisy)],summary[candidate][str(noisy)]
                checks[f'{noisy}_spell']=b['spell_f1']>a['spell_f1'];checks[f'{noisy}_short']=b['short_hits']>=a['short_hits']
        gates[candidate]=checks
        if all(checks.values()):eligible.append(candidate)
    def rank(candidate):return tuple(sum(summary[candidate][str(n)][key] for n in [False,True])/2 for key in ['spell_f1','macro_f1'])
    selected=max(eligible,key=rank) if eligible else None
    report=dict(status='CPU-only frozen-network TTA, reused external development comparison, no competition access',
        baseline_sha256=digest(ROOT/'models/external_v4_1.joblib'),model_hashes=model_hashes,
        source_sha256={name:digest(ROOT/name) for name in ['tta_research.py','tta_features.py','TTA_PROTOCOL.md']},
        summary=summary,gates=gates,selected=selected,accepted=selected is not None,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
