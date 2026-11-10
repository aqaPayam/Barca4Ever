"""Fit a fixed small reliability-rule family on WR9, evaluate other matches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import learned_possession
import joblib
from joblib import Parallel,delayed
from external_idsse import SPLITS
from research_conservative_fusion import data_for,digest
from score_fusion import combine_score
from temporal_state import decode
from develop_v2 import score
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


def calibration_trial(fingerprint,a,b):
    rows=[]
    for h in [0,1]:
        for noisy in [False,True]:
            d=data_for(fingerprint,'J03WR9',h,noisy,8400+h)
            p=combine_score(dict(a=a,b=b),d['tree'],d['neural'],d['f'])
            rows.append(score(d['y'],decode(p,d['onset'],16,8)))
    return dict(a=a,b=b,**aggregate(rows))


def fitted_gate(tag,base_matches):
    target=ROOT/f'models/smart_fusion/score_{tag}.json'
    if target.exists():
        report=json.loads(target.read_text())
        assert report['base_training_matches']==base_matches
        return report
    if tag=='full':fingerprint=hashlib.sha256((digest(ROOT/'models/external_v4_1.joblib')+digest(ROOT/'smart_fusion.py')).encode()).hexdigest()
    else:fingerprint=joblib.load(ROOT/f'models/smart_fusion/joint_{tag}.joblib')['base_fingerprint']
    trials=Parallel(n_jobs=4)(delayed(calibration_trial)(fingerprint,a,b) for a in [-.2,-.1,0,.1,.2] for b in [-.2,-.1,0,.1,.2])
    selected=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1'],-r['a']**2-r['b']**2))
    report=dict(fitting_matches=['J03WR9'],base_training_matches=base_matches,base_fingerprint=fingerprint,
        selected=selected,trials=trials,protocol_sha256=digest(ROOT/'SCORE_FUSION_PROTOCOL.md'),source_sha256=digest(ROOT/'score_fusion.py'))
    target.write_text(json.dumps(report,indent=2)+'\n')
    print('Spell-score calibration fitted:',tag,selected['a'],selected['b'],flush=True)
    return report


def evaluate(tag,match,h,noisy,seed,reference):
    artifact=json.loads((ROOT/f'models/smart_fusion/score_{tag}.json').read_text())
    d=data_for(artifact['base_fingerprint'],match,h,noisy,seed)
    rows=[]
    for name in ['v4.1','score_gate']:
        p=.5*(d['tree']+d['neural']) if name=='v4.1' else combine_score(artifact['selected'],d['tree'],d['neural'],d['f'])
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
    args=parser.parse_args();target=ROOT/f'external_reports/score_fusion_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve score-fusion trial evidence')
    if args.stage=='crosscheck':folds=[('fold1',SPLITS['train'][:2],SPLITS['train'][2:],0),('fold2',SPLITS['train'][2:],SPLITS['train'][:2],1)]
    else:
        assert json.loads((ROOT/'external_reports/score_fusion_crosscheck.json').read_text())['selected'],'CV failed; keep V4.1'
        folds=[('full',SPLITS['train'],SPLITS['test'],None)]
    reference=json.loads((ROOT/('external_reports/v4_decoder_crosscheck.json' if args.stage=='crosscheck' else 'external_reports/v4_decoder_regression.json')).read_text())['selected']['rows']
    lookup={(r['match'],r['half'],r['noisy']):r for r in reference}
    tasks=[]
    for tag,base_matches,matches,fold in folds:
        assert not set(base_matches+['J03WR9'])&set(matches)
        fitted_gate(tag,base_matches)
        for m,match in enumerate(matches):
            for h in [0,1]:
                for noisy in [False,True]:
                    seed=11200+100*fold+10*m+h if fold is not None else 9500+h
                    tasks.append((tag,match,h,noisy,seed,lookup[(match,h+1,noisy)]))
    rows=[r for group in Parallel(n_jobs=4)(delayed(evaluate)(*task) for task in tasks) for r in group]
    summaries={name:{str(noisy):aggregate([r for r in rows if r['model']==name and r['noisy']==noisy]) for noisy in [False,True]} for name in ['v4.1','score_gate']}
    checks={}
    for noisy in [False,True]:
        a,b=summaries['v4.1'][str(noisy)],summaries['score_gate'][str(noisy)]
        checks[f'{noisy}_spell']=b['spell_f1']>a['spell_f1'];checks[f'{noisy}_short']=b['short_hits']>=a['short_hits']
    report=dict(status='External development; fold-specific WR9 spell-score fitting, other matches evaluate; no competition access',
        baseline_sha256=digest(ROOT/'models/external_v4_1.joblib'),protocol_sha256=digest(ROOT/'SCORE_FUSION_PROTOCOL.md'),
        source_sha256=digest(__file__),summary=summaries,gates={'score_gate':checks},selected='score_gate' if all(checks.values()) else None,rows=rows)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
