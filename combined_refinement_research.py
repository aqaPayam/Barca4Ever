"""Predeclared composition of observability emissions and directed boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import learned_possession
import joblib
from hz25_research import observe,digest
from hz25_model import load as load_network
from external_idsse import SPLITS
from boundary_refinement_research import geometry,components
from boundary_refinement import matrix_scores,blend_scores,decode
from develop_v2 import score
from report_v2 import aggregate
from torch_runtime import torch

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','crosscheck','regression'],required=True)
    args=parser.parse_args();target=ROOT/f'external_reports/combined_refinement_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve composite evidence')
    neural_choice=json.loads((ROOT/'external_reports/hz25_selection.json').read_text())['selected']
    assert neural_choice,'No selected neural adaptation'
    rate=neural_choice['rate'];weight=neural_choice['weight']
    assert json.loads((ROOT/'external_reports/boundary_refinement_validation.json').read_text())['passes_validation']
    if args.stage=='validation':
        alphas=[0,.5,1.];folds=[('full',SPLITS['train'],SPLITS['validation'],None)]
        control_file='v4_decoder_validation.json'
    else:
        selection=json.loads((ROOT/'external_reports/combined_refinement_validation.json').read_text())
        assert selection['selected'] is not None,'No qualified composite'
        alphas=[selection['selected']['alpha']]
        if args.stage=='crosscheck':
            folds=[('fold1',SPLITS['train'][:2],SPLITS['train'][2:],0),('fold2',SPLITS['train'][2:],SPLITS['train'][:2],1)]
            control_file='v4_decoder_crosscheck.json'
        else:
            assert json.loads((ROOT/'external_reports/combined_refinement_crosscheck.json').read_text())['accepted']
            folds=[('full',SPLITS['train'],SPLITS['test'],None)];control_file='v4_decoder_regression.json'
    refs=json.loads((ROOT/'external_reports'/control_file).read_text())['selected']['rows']
    lookup={(r.get('match','J03WR9'),r['half'],r['noisy']):r for r in refs}
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    frozen=joblib.load(ROOT/'models/external_v4_1.joblib');rows=[]
    for tag,train,matches,fold in folds:
        suffix='' if tag=='full' else '_'+tag
        path=ROOT/f'models/hz25/rate{rate}{suffix}.pt'
        model,a=load_network(path,device)
        assert a['training_matches']==train and a['rate']==rate
        pair_artifact=joblib.load(ROOT/f'models/boundary_refinement/{tag}.joblib')
        assert pair_artifact['training_matches']==train
        if fold is None:tree,boundary=frozen['tree_model'],frozen['boundary_model']
        else:
            tree=joblib.load(ROOT/f'external_cache/research_fusion_cv/base_{tag}.joblib')
            boundary=joblib.load(ROOT/f'external_cache/blended_boundaries/{tag}.joblib')['state']
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else (8400 if args.stage=='validation' else 9500)+h
                    d=observe(match,h,noisy,seed,rate,tree,boundary,model,device)
                    x,y=geometry(match,h,noisy,seed);assert np.array_equal(y,d['y'])
                    pair=matrix_scores(pair_artifact['model'],x)
                    p=(1-weight)*d['tree']+weight*d['neural']
                    for alpha in alphas:
                        result=score(y,decode(p,blend_scores(d['onset'],pair,alpha),16,8))
                        rows.append(dict(match=match,half=h+1,noisy=noisy,model='candidate',alpha=alpha,**result))
                    old=components(tag,match,h,noisy,seed)
                    control=score(y,decode(old['p'],blend_scores(old['b'],pair,0),16,8))
                    reference=lookup[(match,h+1,noisy)]
                    assert control['spell_f1']==reference['spell_f1'] and control['macro_f1']==reference['macro_f1']
                    assert control['recall_by_duration']==reference['recall_by_duration']
                    rows.append(dict(match=match,half=h+1,noisy=noisy,model='v4.1',alpha=0,**control))
                    print('External composite comparison:',tag,match,h+1,noisy,flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    if args.stage=='validation':
        trials=[dict(alpha=alpha,rate=rate,weight=weight,**aggregate([r for r in rows if r['model']=='candidate' and r['alpha']==alpha])) for alpha in alphas]
        base=aggregate([r for r in rows if r['model']=='v4.1'])
        neural=next(r for r in trials if r['alpha']==0)
        assert abs(neural['spell_f1']-neural_choice['spell_f1'])<1e-12
        eligible=[r for r in trials if r['spell_f1']>max(base['spell_f1'],neural['spell_f1']) and r['short_hits']>=max(base['short_hits'],neural['short_hits'])]
        selected=max(eligible,key=lambda r:(r['spell_f1'],r['macro_f1'])) if eligible else None
        report=dict(status='External reused WR9 composite selection',neural_choice=neural_choice,baseline=base,trials=trials,selected=selected,rows=rows)
    else:
        summary={name:{str(n):aggregate([r for r in rows if r['model']==name and r['noisy']==n]) for n in [False,True]} for name in ['v4.1','candidate']}
        gates={}
        for n in [False,True]:
            old,new=summary['v4.1'][str(n)],summary['candidate'][str(n)]
            gates[f'{n}_spell']=new['spell_f1']>old['spell_f1'];gates[f'{n}_short']=new['short_hits']>=old['short_hits']
        report=dict(status='External development composite comparison, not new blind testing',choice=selection['selected'],summary=summary,gates=gates,accepted=all(gates.values()),rows=rows)
    report['protocol_sha256']=digest(ROOT/'COMBINED_REFINEMENT_PROTOCOL.md');report['source_sha256']=digest(__file__)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
