"""Freeze a decoder-only refinement after all external acceptance checks pass."""
import hashlib
import json
from pathlib import Path
import learned_possession
import joblib

ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    summary=json.loads((ROOT/'external_reports/v4_decoder_summary.json').read_text())
    assert summary['all_gates_passed'] and all(summary['gates'].values())
    oldpath=ROOT/'models/external_v4.joblib'
    target=ROOT/'models/external_v4_1.joblib'
    if target.exists():raise RuntimeError('Preserve frozen refinement')
    meta=json.loads(Path(str(oldpath)+'.metadata.json').read_text())
    assert digest(oldpath)==meta['model_sha256']
    for name,value in meta['code_sha256'].items():assert digest(ROOT/name)==value
    for stage in ['validation','crosscheck','regression']:
        r=json.loads((ROOT/f'external_reports/v4_decoder_{stage}.json').read_text())
        assert r['frozen_model_sha256']==meta['model_sha256']
        assert r['protocol_sha256']==digest(ROOT/'V4_DECODER_PROTOCOL.md')
        assert r['source_sha256']==digest(ROOT/'research_v4_decoder.py')
        assert all(r['selected'][key]==value for key,value in summary['selected'].items())
    artifact=joblib.load(oldpath)
    unchanged={k:joblib.hash(value) for k,value in artifact.items() if k not in ['penalty','boundary_scale']}
    artifact['penalty']=summary['selected']['base']
    artifact['boundary_scale']=summary['selected']['scale']
    joblib.dump(artifact,target,compress=3)
    restored=joblib.load(target)
    for key,value in unchanged.items():assert joblib.hash(restored[key])==value,key
    refined=dict(meta)
    refined.update(version='4.1',parent_model_sha256=meta['model_sha256'],model_sha256=digest(target),
        parent_gates=meta['gates'],gates=summary['gates'],penalty=artifact['penalty'],boundary_scale=artifact['boundary_scale'],
        evidence='Decoder-only external development validation, whole-match comparison and reused-test regression; small gains, not fresh blind testing',
        refinement='All weights unchanged; boundary scale selected from six predefined external validation settings',
        refinement_evidence_sha256={name:digest(ROOT/name) for name in ['V4_DECODER_PROTOCOL.md','research_v4_decoder.py','external_reports/v4_decoder_validation.json','external_reports/v4_decoder_crosscheck.json','external_reports/v4_decoder_regression.json','external_reports/v4_decoder_summary.json']})
    Path(str(target)+'.metadata.json').write_text(json.dumps(refined,indent=2)+'\n')
    print('V4.1 frozen after all external gates passed; all neural/tree weights verified unchanged.')
