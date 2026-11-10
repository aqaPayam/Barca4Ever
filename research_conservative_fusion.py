"""Cached, external-only conservative gate comparisons with frozen V4.1."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import learned_possession
from smart_fusion import fit_gate
from conservative_fusion import blend
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate
from external_idsse import SPLITS
import joblib
from joblib import Parallel,delayed

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_for(fingerprint,match,h,noisy,seed):
    target=ROOT/f'external_cache/smart_fusion/{fingerprint[:16]}_{match}_{h}_{int(noisy)}_{seed}.npz'
    with np.load(target) as z:
        assert str(z['fingerprint'])==fingerprint
        return {k:z[k] for k in ['y','tree','neural','f','onset']}


def evaluate_case(tag,match,h,noisy,seed,candidates,control):
    kinds=sorted({name.split(':')[0] for name in candidates})
    gates={kind:joblib.load(ROOT/f'models/smart_fusion/{kind}_{tag}.joblib') for kind in kinds}
    fingerprint=next(iter(gates.values()))['base_fingerprint']
    assert all(g['base_fingerprint']==fingerprint for g in gates.values())
    data=data_for(fingerprint,match,h,noisy,seed)
    out=[]
    for name in ['v4.1']+candidates:
        if name=='v4.1':p=.5*(data['tree']+data['neural'])
        else:
            kind,variant=name.split(':')
            p=blend(gates[kind],variant,data['tree'],data['neural'],data['f'],data['onset'])
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
        result=score(data['y'],decode(p,data['onset'],16,8))
        if name=='v4.1':
            assert abs(result['spell_f1']-control['spell_f1'])<1e-12
            assert abs(result['macro_f1']-control['macro_f1'])<1e-12
            assert result['recall_by_duration']==control['recall_by_duration']
        out.append(dict(match=match,half=h+1,noisy=noisy,model=name,**result))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['crosscheck','regression'],default='crosscheck')
    args=parser.parse_args()
    target=ROOT/f'external_reports/conservative_fusion_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve experiment evidence')
    candidates=[f'{kind}:{variant}' for kind in ['joint','factorized'] for variant in ['shrink25','shrink50','disagree','boundary']]
    if args.stage=='regression':
        selected=json.loads((ROOT/'external_reports/conservative_fusion_crosscheck.json').read_text())['selected']
        assert selected,'No candidate passed CV; preserve V4.1'
        candidates=[selected]
        kind=selected.split(':')[0]
        path=ROOT/f'models/smart_fusion/{kind}_full.joblib'
        if not path.exists():
            fingerprint=hashlib.sha256((digest(ROOT/'models/external_v4_1.joblib')+digest(ROOT/'smart_fusion.py')).encode()).hexdigest()
            calibration=[data_for(fingerprint,'J03WR9',h,noisy,8400+h) for h in [0,1] for noisy in [False,True]]
            gate=fit_gate(calibration,kind)
            gate.update(fitting_matches=['J03WR9'],base_training_matches=SPLITS['train'],base_fingerprint=fingerprint,
                protocol_sha256=digest(ROOT/'SMART_FUSION_PROTOCOL.md'),source_sha256=digest(ROOT/'smart_fusion.py'))
            joblib.dump(gate,path,compress=3)
    controls=json.loads((ROOT/('external_reports/v4_decoder_crosscheck.json' if args.stage=='crosscheck' else 'external_reports/v4_decoder_regression.json')).read_text())['selected']['rows']
    lookup={(r['match'],r['half'],r['noisy']):r for r in controls}
    tasks=[]
    folds=[('fold1',SPLITS['train'][2:],0),('fold2',SPLITS['train'][:2],1)] if args.stage=='crosscheck' else [('full',SPLITS['test'],None)]
    for tag,matches,fold in folds:
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else 9500+h
                    tasks.append((tag,match,h,noisy,seed,candidates,lookup[(match,h+1,noisy)]))
    results=Parallel(n_jobs=4)(delayed(evaluate_case)(*task) for task in tasks)
    rows=[r for group in results for r in group]
    summaries={name:{str(noisy):aggregate([r for r in rows if r['model']==name and r['noisy']==noisy]) for noisy in [False,True]} for name in ['v4.1']+candidates}
    accepted=[];checks={}
    for name in candidates:
        check={}
        for noisy in [False,True]:
            a,b=summaries['v4.1'][str(noisy)],summaries[name][str(noisy)]
            check[f'{noisy}_spell']=b['spell_f1']>a['spell_f1']
            check[f'{noisy}_short']=b['short_hits']>=a['short_hits']
        checks[name]=check
        if all(check.values()):accepted.append(name)
    def rank(name):return tuple(sum(summaries[name][str(n)][key] for n in [False,True])/2 for key in ['spell_f1','macro_f1'])
    selected=max(accepted,key=rank) if accepted else None
    report=dict(status='External development reuse; CV selects variant, one candidate goes to regression; no competition access',
        baseline_sha256=digest(ROOT/'models/external_v4_1.joblib'),protocol_sha256=digest(ROOT/'SMART_FUSION_CONSERVATIVE_PROTOCOL.md'),
        source_sha256=digest(__file__),summary=summaries,gates=checks,selected=selected,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
