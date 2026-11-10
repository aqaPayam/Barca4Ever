"""Check a saved refined CPU producer on licensed external regression halves."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from external_idsse import load
from learned_possession import corrupt
from develop_v2 import score
from refined_infer import restore_network,predict_block
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def producer_hashes(artifact):
    names=['refined_infer.py','hz25_data.py','contact_input.py','paper2023_data.py',
           'paper2023_model.py','possession_v2.py','learned_possession.py','infer_learned.py',
           'temporal_decoders.py','temporal_state.py','boundary_refinement.py',
           'torch_runtime.py','games.py','barca4ever.py','external_idsse.py']
    if artifact.get('pair_feature_mode')=='physics':names+=['boundary_physics.py','touch_physics.py']
    return {name:digest(ROOT/name) for name in names}


def short_count(result):
    bucket=result['recall_by_duration']['under_2s']
    count=int(bucket['count'])
    return count,round(count*bucket['recall_iou50']) if count else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='models/refined_candidate_combined.joblib')
    parser.add_argument('--reference',default='external_reports/combined_refinement_regression.json')
    parser.add_argument('--out',default='external_reports/refined_production_check.json')
    parser.add_argument('--tolerance',type=float,default=.003)
    args=parser.parse_args()
    if not 0<=args.tolerance<=.003:raise ValueError('F1 tolerance must be between0 and0.003')
    target=ROOT/args.out
    if target.exists():raise RuntimeError('Preserve previous refined producer-check evidence')
    model_path=ROOT/args.model;reference_path=ROOT/args.reference
    model_hash=digest(model_path);artifact=joblib.load(model_path)
    reference=json.loads(reference_path.read_text())
    choice=reference['choice']
    for name,key in [('pair_alpha','alpha'),('rate','rate'),('neural_weight','weight')]:
        assert artifact[name]==choice[key],'Saved artifact does not match the selected method settings'
    before=producer_hashes(artifact)
    torch.set_num_threads(4)
    model=restore_network(artifact,'cpu')
    rows=[];failure=None
    try:
        for match in ['J03WMX','J03WN1']:
            half=load(match)[0]
            for noisy in [False,True]:
                players,ball=half['players'],half['ball']
                if noisy:players,ball=corrupt(players,ball,9500)
                prediction=predict_block(artifact,model,players,half['teams'],ball,half['pitch'],device='cpu')
                result=score(half['labels'],prediction)
                expected=next(r for r in reference['rows'] if r['model']=='candidate' and r['match']==match and r['half']==1 and r['noisy']==noisy)
                count,hits=short_count(result);expected_count,expected_hits=short_count(expected)
                differences=dict(spell_f1=result['spell_f1']-expected['spell_f1'],
                                 macro_f1=result['macro_f1']-expected['macro_f1'],
                                 short_count=count-expected_count,short_hits=hits-expected_hits)
                passed=(abs(differences['spell_f1'])<=args.tolerance and
                        abs(differences['macro_f1'])<=args.tolerance and count==expected_count)
                rows.append(dict(match=match,half=1,noisy=noisy,seed=9500 if noisy else None,
                    actual=result,reference=expected,differences=differences,
                    within_tolerance=passed,short_hits_exact=hits==expected_hits))
                print('External CPU refined-producer comparison completed.',flush=True)
    except Exception as error:
        failure=type(error).__name__
    after=producer_hashes(artifact)
    unchanged=before==after and digest(model_path)==model_hash
    passed=not failure and len(rows)==4 and unchanged and all(r['within_tolerance'] for r in rows)
    report=dict(status='passed' if passed else 'failed',device='cpu',torch_threads=4,
        scope='Licensed external WMX/WN1 first halves, same deterministic corruption; no supplied competition inputs',
        tolerance=args.tolerance,model_sha256=model_hash,model_sha256_after=digest(model_path),
        producer_sha256=before,producer_sha256_after=after,model_and_producer_unchanged=unchanged,
        reference_sha256=digest(reference_path),checker_sha256=digest(__file__),
        all_short_hits_exact=all(r['short_hits_exact'] for r in rows) if rows else False,
        error_type=failure,rows=rows)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise RuntimeError('Saved refined producer failed external CPU verification; inspect its local JSON report')
    print('Saved refined CPU producer passed four external comparisons; differences and hashes recorded.')
