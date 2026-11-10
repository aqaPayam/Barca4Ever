"""Describe frozen V4 errors on external validation only; never fit or promote.

Counts describe overlapping symptoms, not causal attribution or achievable gains.
The only match accepted is the licensed external validation match J03WR9.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from research_data import get as tree_data
from paper2023_data import get as object_data, relative_features
from paper2023_model import outputs, upsample, joint_probabilities
from infer_contact import restore_network
from temporal_decoders import boundary_features
from temporal_state import decode
from develop_v2 import score
from evaluation import possession_metrics as pm
from torch_runtime import torch
import joblib

ROOT = Path(__file__).resolve().parent


def symptoms(y, prediction, tree, neural, mixed):
    runs = pm.valid_runs(y)
    spells = pm.possession_spells(y, runs)
    def flags(p):
        return pm.matched_gt_flags(pm.spell_overlaps(spells, pm.possession_spells(p, runs)), len(spells), .5)
    recovered, raw_recovered = flags(prediction), flags(mixed.argmax(1))
    tree_label, neural_label, mixed_label = tree.argmax(1), neural.argmax(1), mixed.argmax(1)
    counts = dict(count=0, hits=0, no_correct_frames=0, less_than_half_correct=0,
                  majority_correct_but_spell_unmatched=0, raw_spell_lost_after_decoding=0,
                  raw_component_majority_lost_by_mixing=0,
                  raw_mixed_majority_lost_after_decoding=0)
    bins = {name: dict(count=0, hits=0) for name in ['under_half_second', 'half_to_one_second', 'one_to_two_seconds']}
    for i, (a,b,k) in enumerate(spells):
        if b-a >= 50:
            continue
        counts['count'] += 1
        hit = bool(recovered[i])
        counts['hits'] += int(hit)
        bucket = bins['under_half_second' if b-a < 12.5 else 'half_to_one_second' if b-a < 25 else 'one_to_two_seconds']
        bucket['count'] += 1
        bucket['hits'] += int(hit)
        if hit:
            continue
        coverage = float((prediction[a:b] == k).mean())
        name = 'no_correct_frames' if coverage == 0 else 'less_than_half_correct' if coverage < .5 else 'majority_correct_but_spell_unmatched'
        counts[name] += 1
        counts['raw_spell_lost_after_decoding'] += int(raw_recovered[i])
        raw_majority = (mixed_label[a:b] == k).mean() >= .5
        component_majority = max((tree_label[a:b] == k).mean(), (neural_label[a:b] == k).mean()) >= .5
        counts['raw_component_majority_lost_by_mixing'] += int(component_majority and not raw_majority)
        counts['raw_mixed_majority_lost_after_decoding'] += int(raw_majority and coverage < .5)
    assert counts['count']-counts['hits'] == sum(counts[k] for k in ['no_correct_frames','less_than_half_correct','majority_correct_but_spell_unmatched'])
    assert sum(v['count'] for v in bins.values()) == counts['count']
    return dict(**counts, duration_bins=bins)


if __name__ == '__main__':
    target = ROOT/'external_reports/v4_short_error_diagnostic.json'
    if target.exists():
        raise RuntimeError('Preserve existing diagnostic evidence')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = ROOT/'models/external_v4.joblib'
    artifact = joblib.load(model_path)
    assert artifact['role_mode'] == 'none' and artifact['boundary_kind'] == 'state'
    model = restore_network(artifact, device)
    references = json.loads((ROOT/'models/paper2023/fusion_rolefree.validation.json').read_text())['selected']['rows']
    rows = []
    for h in [0,1]:
        for noisy in [False,True]:
            data = tree_data('J03WR9',h,noisy,split='validation')
            objects = object_data('J03WR9',h,noisy,seed=8400+h,interpolation='short')
            assert np.array_equal(data['y'],objects['full_y'])
            objects['x'][:,:,6] = 0
            neural = joint_probabilities(upsample(outputs(model,relative_features(objects['x']),device),objects['samples'],len(data['y'])))
            tree = artifact['tree_model'].predict_proba(data['x'])
            mixed = (1-artifact['neural_weight'])*tree+artifact['neural_weight']*neural
            onset = artifact['boundary_model'].predict_proba(boundary_features(data['x']))[:,1:]
            prediction = decode(mixed,onset,artifact['penalty'],artifact['boundary_scale'])
            scored = score(data['y'],prediction)
            reference = next(r for r in references if r['half']==h+1 and r['noisy']==noisy)
            assert abs(scored['spell_f1']-reference['spell_f1']) < 1e-12, 'Frozen V4 control failed'
            row = dict(half=h+1,noisy=noisy,**symptoms(data['y'],prediction,tree,neural,mixed))
            assert row['hits'] == round(scored['recall_by_duration']['under_2s']['recall_iou50']*row['count'])
            rows.append(row)
            print('External validation diagnostic complete:',h+1,noisy,flush=True)
    aggregate = []
    for noisy in [False,True]:
        group = [r for r in rows if r['noisy']==noisy]
        totals = {k:sum(r[k] for r in group) for k in group[0] if k not in ['half','noisy','duration_bins']}
        totals['duration_bins'] = {k:{v:sum(r['duration_bins'][k][v] for r in group) for v in ['count','hits']} for k in group[0]['duration_bins']}
        aggregate.append(dict(noisy=noisy,**totals))
    report = dict(status='Descriptive overlapping symptoms on reused external validation only; no fitting, causal claims, promotion or supplied-match access',
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,aggregate=aggregate)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(aggregate),flush=True)
