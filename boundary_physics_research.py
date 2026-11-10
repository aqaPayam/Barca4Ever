"""External-only dense boundary training with selected soft contact physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import learned_possession
from external_idsse import load,SPLITS
from learned_possession import corrupt
from research_data import get
from boundary_refinement import targets,blend_scores,decode
from boundary_refinement_research import digest,components,geometry
from boundary_physics import physics_features,combined_features,dense_sampling,matrix_scores
from develop_v2 import score
from report_v2 import aggregate
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models/boundary_physics'
CACHE=ROOT/'external_cache/boundary_physics'
SOURCES=['boundary_physics.py','boundary_refinement.py','touch_physics.py']


def physical(match,h,noisy,seed,swap=False):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/f'{match}_{h}_{int(noisy)}_{seed}_{int(swap)}.npz'
    source_hash=''.join(digest(ROOT/p) for p in SOURCES)
    if path.exists():
        with np.load(path) as z:
            assert str(z['source_hash'])==source_hash
            return z['physics']
    half=load(match)[h];players,ball=half['players'],half['ball']
    if noisy:players,ball=corrupt(players,ball,seed)
    teams=1-half['teams'] if swap else half['teams']
    result=physics_features(players,teams,ball,half['pitch'])
    assert np.isfinite(result).all()
    np.savez_compressed(path,physics=result,source_hash=source_hash)
    return result


def fit(matches,tag):
    MODELS.mkdir(parents=True,exist_ok=True);path=MODELS/f'{tag}.joblib'
    source_hashes={p:digest(ROOT/p) for p in SOURCES}
    if path.exists():
        artifact=joblib.load(path)
        assert artifact['training_matches']==matches and artifact['source_sha256']==source_hashes
        return artifact['model']
    xx,yy,ww=[],[],[]
    for match in matches:
        for h in [0,1]:
            seed=7300+10*SPLITS['train'].index(match)+h
            for noisy in [False,True]:
                for swap in [False,True]:
                    d=get(match,h,noisy,swap)
                    take,weights=dense_sampling(d['y'])
                    f=combined_features(d['x'],physical(match,h,noisy,seed,swap))
                    xx.append(f[take]);yy.append(targets(d['y'])[take]);ww.append(weights[take])
        print('Prepared external dense physics training:',tag,match,flush=True)
    x,y,w=map(np.concatenate,[xx,yy,ww])
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=11,learning_rate=.08,
        l2_regularization=5,min_samples_leaf=80,early_stopping=False,random_state=20261116)
    model.fit(x,y,sample_weight=w)
    assert model.classes_.tolist()==list(range(7))
    joblib.dump(dict(model=model,training_matches=matches,source_sha256=source_hashes,
        protocol_sha256=digest(ROOT/'boundary_physics_protocol.md')),path,compress=3)
    print('Fitted external dense physics classifier:',tag,flush=True)
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','crosscheck','regression'],default='validation')
    args=parser.parse_args();target=ROOT/f'external_reports/boundary_physics_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve physics-boundary evidence')
    if args.stage=='validation':
        alphas=[0,.25,.5,1.];folds=[('full',SPLITS['train'],SPLITS['validation'],None)]
        control_file='v4_decoder_validation.json'
    else:
        choice=json.loads((ROOT/'external_reports/boundary_physics_validation.json').read_text())
        assert choice['passes_validation'],'Physics validation did not qualify'
        alphas=[0,choice['selected']['alpha']]
        if args.stage=='crosscheck':
            folds=[('fold1',SPLITS['train'][:2],SPLITS['train'][2:],0),('fold2',SPLITS['train'][2:],SPLITS['train'][:2],1)]
            control_file='v4_decoder_crosscheck.json'
        else:
            assert json.loads((ROOT/'external_reports/boundary_physics_crosscheck.json').read_text())['all_gates_passed'],'Physics CV did not qualify'
            folds=[('full',SPLITS['train'],SPLITS['test'],None)];control_file='v4_decoder_regression.json'
    reference=json.loads((ROOT/'external_reports'/control_file).read_text())['selected']['rows']
    lookup={(r.get('match','J03WR9'),r['half'],r['noisy']):r for r in reference};rows=[]
    for tag,train,matches,fold in folds:
        assert not set(train)&set(matches)
        model=fit(train,tag)
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else (8400 if args.stage=='validation' else 9500)+h
                    x,y=geometry(match,h,noisy,seed);d=components(tag,match,h,noisy,seed)
                    assert np.array_equal(y,d['y'])
                    f=combined_features(x,physical(match,h,noisy,seed))
                    pair=matrix_scores(model,f)
                    for alpha in alphas:
                        result=score(y,decode(d['p'],blend_scores(d['b'],pair,alpha)))
                        if alpha==0:
                            control=lookup[(match,h+1,noisy)]
                            assert result['spell_f1']==control['spell_f1'] and result['macro_f1']==control['macro_f1']
                            assert result['recall_by_duration']==control['recall_by_duration']
                        rows.append(dict(match=match,half=h+1,noisy=noisy,alpha=alpha,**result))
                    print('External dense physics comparison:',tag,match,h+1,noisy,flush=True)
    trials=[dict(alpha=alpha,**aggregate([r for r in rows if r['alpha']==alpha])) for alpha in alphas]
    baseline=trials[0];selected=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1'])) if args.stage=='validation' else trials[1]
    if args.stage=='validation':gates=dict(validation_spell=selected['spell_f1']>baseline['spell_f1'],validation_short=selected['short_hits']>=baseline['short_hits'])
    else:
        gates={}
        for noisy in [False,True]:
            a=aggregate([r for r in rows if r['alpha']==0 and r['noisy']==noisy])
            b=aggregate([r for r in rows if r['alpha']==selected['alpha'] and r['noisy']==noisy])
            gates[f'{noisy}_spell']=b['spell_f1']>a['spell_f1'];gates[f'{noisy}_short']=b['short_hits']>=a['short_hits']
    passes=all(gates.values())
    report=dict(status='External development dense physics transition candidate; no competition access',
        baseline_model_sha256=digest(ROOT/'models/external_v4_1.joblib'),source_sha256=digest(__file__),
        protocol_sha256=digest(ROOT/'boundary_physics_protocol.md'),trials=trials,selected=selected,
        baseline=baseline,gates=gates,passes_validation=passes if args.stage=='validation' else None,all_gates_passed=passes,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
