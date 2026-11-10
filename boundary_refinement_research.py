"""External-only transition-pair boundary fitting and matched comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import learned_possession
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import features
from research_data import get
from develop_v2 import score
from report_v2 import aggregate
from boundary_refinement import temporal_features,targets,matrix_scores,blend_scores,decode
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models/boundary_refinement'
CACHE=ROOT/'external_cache/boundary_refinement'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(matches,tag):
    MODELS.mkdir(parents=True,exist_ok=True)
    path=MODELS/f'{tag}.joblib'
    if path.exists():
        a=joblib.load(path)
        assert a['training_matches']==matches
        assert a['source_sha256']==digest(ROOT/'boundary_refinement.py')
        return a['model']
    xx,yy=[],[]
    for match in matches:
        for h in [0,1]:
            for noisy in [False,True]:
                for swap in [False,True]:
                    d=get(match,h,noisy,swap)
                    take=(np.arange(len(d['y']))%5==0)&(d['y']<3)
                    xx.append(temporal_features(d['x'])[take]);yy.append(targets(d['y'])[take])
        print('Prepared external transition training:',tag,match,flush=True)
    x,y=np.concatenate(xx),np.concatenate(yy)
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=11,learning_rate=.08,
        l2_regularization=5,min_samples_leaf=80,early_stopping=False,random_state=20261116)
    model.fit(x,y,sample_weight=np.where(y,20.,1.))
    assert model.classes_.tolist()==list(range(7))
    artifact=dict(model=model,training_matches=matches,source_sha256=digest(ROOT/'boundary_refinement.py'),
        protocol_sha256=digest(ROOT/'boundary_refinement_protocol.md'))
    joblib.dump(artifact,path,compress=3)
    print('Fitted external transition classifier:',tag,flush=True)
    return model


def geometry(match,h,noisy,seed):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/f'{match}_{h}_{int(noisy)}_{seed}.npz'
    if not path.exists():
        half=load(match)[h];p,b=half['players'],half['ball']
        if noisy:p,b=corrupt(p,b,seed)
        np.savez_compressed(path,x=features(p,half['teams'],b,half['pitch']),y=half['labels'])
    with np.load(path) as z:return z['x'],z['y']


def components(tag,match,h,noisy,seed):
    if tag=='full':fingerprint=digest(ROOT/'models/external_v4.joblib')
    else:
        files=[ROOT/f'models/paper2023/rolefree_{tag}.pt',
               ROOT/f'external_cache/research_fusion_cv/base_{tag}.joblib',
               ROOT/f'external_cache/blended_boundaries/{tag}.joblib']
        fingerprint=hashlib.sha256(''.join(digest(p) for p in files).encode()).hexdigest()
    path=ROOT/f'external_cache/v4_decoder/{fingerprint[:16]}_{match}_{h}_{int(noisy)}_{seed}.npz'
    with np.load(path) as z:
        assert str(z['fingerprint'])==fingerprint
        return {k:z[k] for k in ['y','p','b']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','crosscheck','regression'],default='validation')
    args=parser.parse_args();target=ROOT/f'external_reports/boundary_refinement_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve boundary refinement evidence')
    if args.stage=='validation':
        alphas=[0,.25,.5,1.];folds=[('full',SPLITS['train'],SPLITS['validation'],None)]
        control_file='v4_decoder_validation.json'
    else:
        choice=json.loads((ROOT/'external_reports/boundary_refinement_validation.json').read_text())
        assert choice['passes_validation'],'Validation did not qualify'
        alphas=[0,choice['selected']['alpha']]
        if args.stage=='crosscheck':
            folds=[('fold1',SPLITS['train'][:2],SPLITS['train'][2:],0),('fold2',SPLITS['train'][2:],SPLITS['train'][:2],1)]
            control_file='v4_decoder_crosscheck.json'
        else:
            assert json.loads((ROOT/'external_reports/boundary_refinement_crosscheck.json').read_text())['all_gates_passed'],'CV did not qualify'
            folds=[('full',SPLITS['train'],SPLITS['test'],None)];control_file='v4_decoder_regression.json'
    reference=json.loads((ROOT/'external_reports'/control_file).read_text())['selected']['rows']
    lookup={(r.get('match','J03WR9'),r['half'],r['noisy']):r for r in reference}
    rows=[]
    for tag,train,matches,fold in folds:
        assert not set(train)&set(matches)
        model=fit(train,tag)
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else (8400 if args.stage=='validation' else 9500)+h
                    x,y=geometry(match,h,noisy,seed);d=components(tag,match,h,noisy,seed)
                    assert np.array_equal(y,d['y'])
                    pair=matrix_scores(model,x)
                    for alpha in alphas:
                        result=score(y,decode(d['p'],blend_scores(d['b'],pair,alpha)))
                        if alpha==0:
                            control=lookup[(match,h+1,noisy)]
                            assert result['spell_f1']==control['spell_f1'] and result['macro_f1']==control['macro_f1']
                            assert result['recall_by_duration']==control['recall_by_duration']
                        rows.append(dict(match=match,half=h+1,noisy=noisy,alpha=alpha,**result))
                    print('External transition comparison:',tag,match,h+1,noisy,flush=True)
    trials=[dict(alpha=alpha,**aggregate([r for r in rows if r['alpha']==alpha])) for alpha in alphas]
    baseline=trials[0];selected=max(trials,key=lambda t:(t['spell_f1'],t['macro_f1'])) if args.stage=='validation' else trials[1]
    if args.stage=='validation':
        passes=selected['spell_f1']>baseline['spell_f1'] and selected['short_hits']>=baseline['short_hits']
        gates=dict(validation_spell=selected['spell_f1']>baseline['spell_f1'],validation_short=selected['short_hits']>=baseline['short_hits'])
    else:
        gates={}
        for noisy in [False,True]:
            a=aggregate([r for r in rows if r['alpha']==0 and r['noisy']==noisy])
            b=aggregate([r for r in rows if r['alpha']==selected['alpha'] and r['noisy']==noisy])
            gates[f'{noisy}_spell']=b['spell_f1']>a['spell_f1'];gates[f'{noisy}_short']=b['short_hits']>=a['short_hits']
        passes=all(gates.values())
    report=dict(status='External development transition-pair refinement; no competition access',
        baseline_model_sha256=digest(ROOT/'models/external_v4_1.joblib'),source_sha256=digest(__file__),
        protocol_sha256=digest(ROOT/'boundary_refinement_protocol.md'),trials=trials,selected=selected,
        baseline=baseline,gates=gates,passes_validation=passes if args.stage=='validation' else None,all_gates_passed=passes,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
