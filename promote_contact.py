"""Package a contact fusion only after complete external acceptance checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from report_v2 import aggregate
from torch_runtime import torch
import learned_possession  # project-local sklearn/joblib dependencies
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    choice=json.loads((ROOT/'models/paper2023/fusion_rolefree.validation.json').read_text())
    cross=json.loads((ROOT/'external_reports/paper2023_rolefree_crosscheck.json').read_text())['rows']
    regression=json.loads((ROOT/'external_reports/paper2023_regression_fusion_rolefree.json').read_text())['rows']
    oldreg=json.loads((ROOT/'external_reports/research_v3/boundary_regression.json').read_text())['rows']
    oldval=json.loads((ROOT/'external_reports/research_v3/learned_boundary.json').read_text())['selected']
    oldval_summary=aggregate(oldval['rows']);newval_summary=aggregate(choice['selected']['rows'])
    gates=dict(validation_spell=newval_summary['spell_f1']>oldval_summary['spell_f1'],
               validation_short=newval_summary['short_hits']>=oldval_summary['short_hits'])
    for noisy in [False,True]:
        for dataset,new,old,oldname in [('cross',cross,cross,'v3'),('regression',regression,oldreg,'learned_boundary')]:
            a=aggregate([r for r in old if r['model']==oldname and r['noisy']==noisy])
            b=aggregate([r for r in new if r['model']=='fusion' and r['noisy']==noisy])
            gates[f'{dataset}_{noisy}_spell']=b['spell_f1']>a['spell_f1']
            gates[f'{dataset}_{noisy}_short']=b['short_hits']>=a['short_hits']
    if not all(gates.values()):raise RuntimeError('External acceptance checks did not pass: '+str(gates))
    neural_path=ROOT/'models/paper2023/rolefree.pt'
    neural_hash=hashlib.sha256(neural_path.read_bytes()).hexdigest()
    assert neural_hash==choice['neural_sha256']
    neural=torch.load(neural_path,map_location='cpu',weights_only=True)
    assert neural['recipe']=='rolefree' and neural['run_info']['role_mode']=='none'
    v3=joblib.load(ROOT/'models/external_v3.joblib');state=joblib.load(ROOT/'models/state_boundary.joblib')
    selected=choice['selected']
    artifact=dict(kind='contact_fusion',role_mode='none',architecture=neural['architecture'],feature_version=2,
        neural_state={k:v.cpu().numpy().copy() for k,v in neural['state_dict'].items()},
        tree_model=v3['model'],boundary_kind=selected['boundary_kind'],
        boundary_model=v3['boundary_model'] if selected['boundary_kind']=='binary' else state['boundary_model'],
        neural_weight=selected['neural_weight'],penalty=16,boundary_scale=4)
    target=ROOT/'models/external_v4.joblib'
    if target.exists():raise RuntimeError('Preserve previously frozen artifact')
    joblib.dump(artifact,target,compress=3)
    restored=joblib.load(target)
    assert restored['architecture']==neural['architecture']
    for key,value in neural['state_dict'].items():
        assert np.array_equal(restored['neural_state'][key],value.cpu().numpy()),key
    assert restored['tree_model'].classes_.tolist()==[0,1,2]
    expected=[0,1] if selected['boundary_kind']=='binary' else [0,1,2,3]
    assert restored['boundary_model'].classes_.tolist()==expected
    sources=['infer_contact.py','contact_input.py','paper2023_data.py','paper2023_model.py',
        'torch_runtime.py','possession_v2.py','learned_possession.py','infer_learned.py',
        'temporal_state.py','temporal_decoders.py','barca4ever.py','games.py']
    meta=dict(kind='contact_fusion',role_mode='none',model_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        original_neural_sha256=neural_hash,neural_weight=selected['neural_weight'],boundary_kind=selected['boundary_kind'],
        gates=gates,training_matches=neural['training_matches'],licence='CC BY 4.0',
        attribution='IDSSE, DFL/Sportec; Bassek et al. 2025; DOI 10.6084/m9.figshare.28196177',
        inference_device='cpu',torch_version='2.9.1',
        architecture=neural['architecture'],
        evidence='External development validation, cross-validation and reused-test regression, not fresh blind testing',
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    Path(str(target)+'.metadata.json').write_text(json.dumps(meta,indent=2))
    print('Contact fusion frozen using external evidence. Earlier artifacts and outputs preserved.')
