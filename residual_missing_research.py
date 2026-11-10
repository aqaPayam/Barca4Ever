"""Pareto-anchor external comparisons for protected missing-input residuals."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from external_idsse import SPLITS
from residual_missing_data import get
from residual_missing_model import load as load_network
from paper2023_model import outputs,upsample,joint_probabilities
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component_fingerprint(tag):
    if not tag:return hashlib.sha256((digest(ROOT/'models/external_v4_1.joblib')+digest(ROOT/'smart_fusion.py')).encode()).hexdigest()
    return joblib.load(ROOT/f'models/smart_fusion/joint{tag}.joblib')['base_fingerprint']


def observations(match,h,noisy,seed,model,device,fingerprint):
    path=ROOT/f'external_cache/smart_fusion/{fingerprint[:16]}_{match}_{h}_{int(noisy)}_{seed}.npz'
    with np.load(path) as z:
        assert str(z['fingerprint'])==fingerprint
        data={key:z[key] for key in ['y','tree','neural','onset']}
    d=get(match,h,noisy,seed)
    assert np.array_equal(d['full_y'],data['y'])
    data['candidate']=joint_probabilities(upsample(outputs(model,d['x'],device),d['samples'],len(data['y'])))
    data['all_extra_channels_zero']=not bool(d['x'][:,:,17:].any())
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','crosscheck','regression'],required=True)
    parser.add_argument('--batch',type=int,choices=[128,256],default=256)
    args=parser.parse_args();target=ROOT/f'external_reports/residual_missing_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve residual comparison evidence')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.stage=='validation':
        reference_path=ROOT/'external_reports/v4_decoder_validation.json'
        folds=[(SPLITS['train'],['J03WR9'],None)]
    else:
        assert json.loads((ROOT/'external_reports/residual_missing_validation.json').read_text())['accepted'],'Residual WR9 failed'
        if args.stage=='regression':assert json.loads((ROOT/'external_reports/residual_missing_crosscheck.json').read_text())['accepted'],'Residual CV failed'
        reference_path=ROOT/('external_reports/v4_decoder_crosscheck.json' if args.stage=='crosscheck' else 'external_reports/v4_decoder_regression.json')
        folds=[(SPLITS['train'][:2],SPLITS['train'][2:],0),(SPLITS['train'][2:],SPLITS['train'][:2],1)] if args.stage=='crosscheck' else [(SPLITS['train'],SPLITS['test'],None)]
    controls=json.loads(reference_path.read_text())['selected']['rows']
    lookup={(r.get('match','J03WR9'),r['half'],r['noisy']):r for r in controls}
    rows=[];models={};zero_extra_cases=[]
    for fit_matches,matches,fold in folds:
        assert not set(fit_matches)&set(matches)
        tag='' if fold is None else f'_fold{fold+1}'
        path=ROOT/f'models/residual_missing/{"full" if fold is None else f"fold{fold+1}"}.pt'
        parent=ROOT/f'models/paper2023/rolefree{tag}.pt'
        if not path.exists():
            assert fold is not None,'Fit the full fixed residual before validation'
            subprocess.run([sys.executable,str(ROOT/'residual_missing_train.py'),'--matches',*fit_matches,
                '--initialize',str(parent),'--out',str(path),'--batch',str(args.batch)],check=True,cwd=ROOT)
        model,a=load_network(path,device)
        assert a['training_matches']==fit_matches and a['epochs']==2 and a['role_mode']=='none'
        assert a['initializer_sha256']==digest(parent)
        assert a['final_proof']['all_old_tensors_bit_exact']
        assert a['final_proof']['before_sha256']==a['final_proof']['after_sha256']
        for name,value in a['source_sha256'].items():assert digest(ROOT/name)==value
        models[tag or 'full']=dict(sha256=digest(path),proof=a['final_proof'],batch_size=a['batch_size'])
        fingerprint=component_fingerprint(tag)
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else (8400 if args.stage=='validation' else 9500)+h
                    d=observations(match,h,noisy,seed,model,device,fingerprint)
                    if d['all_extra_channels_zero']:zero_extra_cases.append(dict(match=match,half=h+1,noisy=noisy))
                    for name,neural in [('v4.1',d['neural']),('candidate',d['candidate'])]:
                        result=score(d['y'],decode(.5*(d['tree']+neural),d['onset'],16,8))
                        if name=='v4.1':
                            ref=lookup[(match,h+1,noisy)]
                            assert result['spell_f1']==ref['spell_f1'] and result['macro_f1']==ref['macro_f1']
                            assert result['recall_by_duration']==ref['recall_by_duration']
                        rows.append(dict(match=match,half=h+1,noisy=noisy,model=name,**result))
                    print('External protected-residual comparison:',match,h+1,noisy,flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    summary={name:{str(noisy):aggregate([r for r in rows if r['model']==name and r['noisy']==noisy]) for noisy in [False,True]} for name in ['v4.1','candidate']}
    mixed={name:aggregate([r for r in rows if r['model']==name]) for name in ['v4.1','candidate']}
    gates={}
    for noisy in [False,True]:
        old,new=summary['v4.1'][str(noisy)],summary['candidate'][str(noisy)]
        gates[f'{noisy}_spell']=new['spell_f1']>old['spell_f1'] if noisy else new['spell_f1']>=old['spell_f1']
        gates[f'{noisy}_short']=new['short_hits']>=old['short_hits']
    accepted=all(gates.values())
    report=dict(status='New Pareto-anchor residual experiment; reused external development only, no competition access',
        input_mode='missing20',rate=16,weight=.5,baseline_sha256=digest(ROOT/'models/external_v4_1.joblib'),
        protocol_sha256=digest(ROOT/'RESIDUAL_MISSING_PROTOCOL.md'),source_sha256=digest(__file__),
        model_sha256=models.get('full',{}).get('sha256'),models=models,summary=summary,mixed=mixed,gates=gates,
        accepted=accepted,selected=dict(input_mode='missing20',rate=16,weight=.5,**mixed['candidate']) if accepted else None,
        all_extra_zero_cases=zero_extra_cases,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
