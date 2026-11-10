"""Promote only on explicit external checks; never inspect competition labels.

These are development acceptance checks, not a preregistered blind-test protocol.
"""
import hashlib
import json
from pathlib import Path
import shutil
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    reports=ROOT/'external_reports'
    val=json.loads((reports/'v2_validation.json').read_text())
    oldval=json.loads((reports/'v1_short_spell_validation.json').read_text())
    cross=json.loads((reports/'v2_crosscheck.json').read_text())
    regression=json.loads((reports/'v2_regression.json').read_text())
    gates={}
    a,b=aggregate(oldval),aggregate(val['selected']['rows'])
    gates['validation_overall']=b['spell_f1']>a['spell_f1']
    gates['validation_short_recall']=b['short_recall']>a['short_recall']
    for name,data in [('cross',cross),('regression',regression)]:
        for noisy in [False,True]:
            a=aggregate([r for r in data['rows'] if r['model']=='v1' and r['noisy']==noisy])
            b=aggregate([r for r in data['rows'] if r['model']=='v2' and r['noisy']==noisy])
            gates['{}_{}_overall'.format(name,noisy)]=b['spell_f1']>a['spell_f1']
            if name=='cross':
                gates['cross_{}_short_recall'.format(noisy)]=b['short_recall']>a['short_recall']
    if not all(gates.values()):
        print(json.dumps(gates))
        raise RuntimeError('External promotion gates not met; v1 retained')
    source=ROOT/'models/external_v2_experimental.joblib'
    if digest(source)!=val['artifact_sha256']:
        raise ValueError('Artifact changed after validation selection')
    dest=ROOT/'models/external_v2.joblib'
    shutil.copyfile(source,dest)
    manifest=dict(model_sha256=digest(dest),feature_version=2,penalty=val['selected']['penalty'],
                  gates=gates,training_matches=['J03WOH','J03WOY','J03WPY','J03WQQ'],
                  validation_match='J03WR9',licence='CC BY 4.0',
                  evidence='Validation, whole-match cross-validation and reused-test regression; no fresh blind test',
                  code_sha256={n:digest(ROOT/n) for n in ['possession_v2.py','learned_possession.py','develop_v2.py','infer_learned.py']})
    (ROOT/'models/external_v2.joblib.metadata.json').write_text(json.dumps(manifest,indent=2))
    # Preserve the previous actual submission before replacing it with v2 inference.
    for suffix in ['', '.manifest.json', '.validation.txt']:
        old=ROOT/('evaluation/results/Barca4Ever.csv'+suffix)
        backup=ROOT/('evaluation/results/Barca4Ever_v1.csv'+suffix)
        if old.exists() and not backup.exists():shutil.copyfile(old,backup)
    print('All external promotion gates passed. V2 artifact frozen; previous submission preserved.')
