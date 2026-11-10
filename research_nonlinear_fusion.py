"""Evaluate nonlinear gates on the existing, matched external component cache."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import learned_possession
import joblib
from joblib import Parallel,delayed
from external_idsse import SPLITS
from nonlinear_fusion import fit_regret,combine_regret
from research_conservative_fusion import data_for,digest
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


def evaluate(tag,match,h,noisy,seed,candidates,reference):
    a=joblib.load(ROOT/f'models/smart_fusion/nonlinear_{tag}.joblib')
    d=data_for(a['base_fingerprint'],match,h,noisy,seed)
    rows=[]
    for name in ['v4.1']+candidates:
        p=.5*(d['tree']+d['neural']) if name=='v4.1' else combine_regret(a['model'],.25 if name=='regret25' else .4,d['tree'],d['neural'],d['f'])
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
        result=score(d['y'],decode(p,d['onset'],16,8))
        if name=='v4.1':
            assert abs(result['spell_f1']-reference['spell_f1'])<1e-12
            assert abs(result['macro_f1']-reference['macro_f1'])<1e-12
            assert result['recall_by_duration']==reference['recall_by_duration']
        rows.append(dict(match=match,half=h+1,noisy=noisy,model=name,**result))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['crosscheck','regression'],default='crosscheck')
    args=parser.parse_args()
    target=ROOT/f'external_reports/nonlinear_fusion_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve nonlinear trial evidence')
    if args.stage=='crosscheck':
        candidates=['regret25','regret40'];folds=[('fold1',SPLITS['train'][2:],0),('fold2',SPLITS['train'][:2],1)]
    else:
        selected=json.loads((ROOT/'external_reports/nonlinear_fusion_crosscheck.json').read_text())['selected']
        assert selected,'Neither nonlinear gate passed; retain V4.1'
        candidates=[selected];folds=[('full',SPLITS['test'],None)]
    reference=json.loads((ROOT/('external_reports/v4_decoder_crosscheck.json' if args.stage=='crosscheck' else 'external_reports/v4_decoder_regression.json')).read_text())['selected']['rows']
    lookup={(r['match'],r['half'],r['noisy']):r for r in reference}
    tasks=[]
    for tag,matches,fold in folds:
        target_model=ROOT/f'models/smart_fusion/nonlinear_{tag}.joblib'
        if not target_model.exists():
            if fold is None:
                fingerprint=hashlib.sha256((digest(ROOT/'models/external_v4_1.joblib')+digest(ROOT/'smart_fusion.py')).encode()).hexdigest()
                base_matches=SPLITS['train']
            else:
                parent=joblib.load(ROOT/f'models/smart_fusion/joint_{tag}.joblib')
                fingerprint=parent['base_fingerprint'];base_matches=parent['base_training_matches']
            assert not set(base_matches+['J03WR9'])&set(matches)
            calibration=[data_for(fingerprint,'J03WR9',h,noisy,8400+h) for h in [0,1] for noisy in [False,True]]
            model=fit_regret(calibration)
            joblib.dump(dict(model=model,fitting_matches=['J03WR9'],base_training_matches=base_matches,
                base_fingerprint=fingerprint,source_sha256=digest(ROOT/'nonlinear_fusion.py'),
                protocol_sha256=digest(ROOT/'NONLINEAR_FUSION_PROTOCOL.md')),target_model,compress=3)
            print('External nonlinear gate fitted:',tag,flush=True)
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else 9500+h
                    tasks.append((tag,match,h,noisy,seed,candidates,lookup[(match,h+1,noisy)]))
    results=Parallel(n_jobs=4)(delayed(evaluate)(*task) for task in tasks)
    rows=[r for group in results for r in group]
    summaries={name:{str(noisy):aggregate([r for r in rows if r['model']==name and r['noisy']==noisy]) for noisy in [False,True]} for name in ['v4.1']+candidates}
    gates={};accepted=[]
    for name in candidates:
        check={}
        for noisy in [False,True]:
            a,b=summaries['v4.1'][str(noisy)],summaries[name][str(noisy)]
            check[f'{noisy}_spell']=b['spell_f1']>a['spell_f1'];check[f'{noisy}_short']=b['short_hits']>=a['short_hits']
        gates[name]=check
        if all(check.values()):accepted.append(name)
    def rank(name):return tuple(sum(summaries[name][str(n)][key] for n in [False,True])/2 for key in ['spell_f1','macro_f1'])
    selected=max(accepted,key=rank) if accepted else None
    report=dict(status='External development CV selection, only one candidate may confirm on reused regression; no competition access',
        baseline_sha256=digest(ROOT/'models/external_v4_1.joblib'),protocol_sha256=digest(ROOT/'NONLINEAR_FUSION_PROTOCOL.md'),
        source_sha256=digest(__file__),summary=summaries,gates=gates,selected=selected,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
