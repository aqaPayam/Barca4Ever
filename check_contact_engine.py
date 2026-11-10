"""Verify the actual production prediction function on an external half only."""
import argparse
import json
from pathlib import Path
from external_idsse import load
from learned_possession import corrupt
from infer_contact import restore_network,predict_block
from develop_v2 import score
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='models/external_v4.joblib')
    parser.add_argument('--reference',default='models/paper2023/fusion_rolefree.validation.json')
    parser.add_argument('--out',default='external_reports/contact_production_check.json')
    args=parser.parse_args()
    target=ROOT/args.out
    if target.exists():raise RuntimeError('Preserve previous production-path evidence')
    torch.set_num_threads(4)
    artifact=joblib.load(ROOT/args.model);model=restore_network(artifact,'cpu')
    half=load('J03WR9')[0]
    reference=json.loads((ROOT/args.reference).read_text())['selected']['rows']
    rows=[]
    for noisy in [False,True]:
        p,b=half['players'],half['ball']
        if noisy:p,b=corrupt(p,b,8400)
        prediction=predict_block(artifact,model,p,half['teams'],b,half['pitch'])
        result=score(half['labels'],prediction)
        expected=next(r for r in reference if r['half']==1 and r['noisy']==noisy)
        if abs(result['spell_f1']-expected['spell_f1'])>.003:
            raise RuntimeError('Production CPU engine differs materially from externally scored research path')
        rows.append(dict(noisy=noisy,**result));print('External CPU production-path check passed',noisy,flush=True)
    target.write_text(json.dumps(dict(status='External CPU production path versus selected GPU research reference, tolerance0.003 spellF1',rows=rows),indent=2))
