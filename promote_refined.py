"""Freeze a paper-informed candidate only after all predeclared quality gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import learned_possession
import joblib
from external_idsse import SPLITS
from torch_runtime import torch

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['neural','directed','combined','physics'],required=True)
    parser.add_argument('--out',default='models/external_v5.joblib')
    args=parser.parse_args()
    target=(ROOT/args.out).resolve()
    assert target.parent==(ROOT/'models').resolve(),'Frozen artifacts stay in the models directory'
    if target.exists():raise RuntimeError('Preserve previously frozen V5')
    prefix=dict(neural='hz25',directed='boundary_refinement',combined='combined_refinement',physics='boundary_physics')[args.method]
    cross=json.loads((ROOT/f'external_reports/{prefix}_crosscheck.json').read_text())
    assert cross.get('accepted',cross.get('all_gates_passed',False)),'CV quality gate failed'
    regression=json.loads((ROOT/f'external_reports/{prefix}_regression.json').read_text())
    for report in [cross,regression]:
        assert report.get('accepted',report.get('all_gates_passed',False)),'Quality gate failed'
        assert all(report['gates'].values())
    original=ROOT/'models/external_v4_1.joblib'
    parent_meta=json.loads(Path(str(original)+'.metadata.json').read_text())
    assert digest(original)==parent_meta['model_sha256']
    for name,value in parent_meta['code_sha256'].items():assert digest(ROOT/name)==value
    parent=joblib.load(original)
    artifact=dict(parent,kind='refined_contact',input_mode='legacy17',rate=16,
                  role_mode='none',pair_alpha=0.,pair_feature_mode='signed')
    evidence=[f'external_reports/{prefix}_crosscheck.json',f'external_reports/{prefix}_regression.json']
    if args.method in ['neural','combined']:
        selection=json.loads((ROOT/'external_reports/hz25_selection.json').read_text())['selected']
        assert selection
        path=ROOT/f"models/hz25/rate{selection['rate']}.pt"
        validation=json.loads((ROOT/f"external_reports/hz25_validation_{selection['rate']}.json").read_text())
        assert validation['model_sha256']==digest(path)
        neural=torch.load(path,map_location='cpu',weights_only=True)
        assert neural['training_matches']==SPLITS['train'] and neural['epochs']==2 and neural['role_mode']=='none'
        assert neural['initializer_sha256']==digest(ROOT/'models/paper2023/rolefree.pt')
        for name,value in neural['source_sha256'].items():assert digest(ROOT/name)==value
        artifact.update(input_mode='confidence20',rate=selection['rate'],neural_weight=selection['weight'],architecture=neural['architecture'],
            neural_state={k:v.numpy().copy() for k,v in neural['state_dict'].items()})
        evidence+=['HZ25_PROTOCOL.md','external_reports/hz25_selection.json',f"external_reports/hz25_validation_{selection['rate']}.json",str(path.relative_to(ROOT))]
    if args.method in ['directed','combined','physics']:
        pair_prefix='boundary_physics' if args.method=='physics' else 'boundary_refinement'
        pair_path=ROOT/f'models/{pair_prefix}/full.joblib'
        pair=joblib.load(pair_path)
        assert pair['training_matches']==SPLITS['train']
        if isinstance(pair['source_sha256'],dict):
            for name,value in pair['source_sha256'].items():assert digest(ROOT/name)==value
        else:assert digest(ROOT/'boundary_refinement.py')==pair['source_sha256']
        validation_path=ROOT/f'external_reports/{prefix}_validation.json'
        validation=json.loads(validation_path.read_text())
        choice=validation['selected'];assert choice
        artifact.update(pair_boundary_model=pair['model'],pair_alpha=choice['alpha'],
                        pair_feature_mode='physics' if args.method=='physics' else 'signed')
        assert artifact['pair_boundary_model'].classes_.tolist()==list(range(7))
        evidence+=[str(pair_path.relative_to(ROOT)),str(validation_path.relative_to(ROOT))]
    if args.method=='combined':evidence+=['COMBINED_REFINEMENT_PROTOCOL.md','boundary_refinement_protocol.md']
    elif args.method=='directed':evidence+=['boundary_refinement_protocol.md']
    elif args.method=='physics':evidence+=['boundary_physics_protocol.md','TOUCH_PHYSICS_PROTOCOL.md']
    sources=['refined_infer.py','hz25_data.py','paper2023_data.py','paper2023_model.py','contact_input.py',
        'boundary_refinement.py','temporal_state.py','temporal_decoders.py','possession_v2.py','learned_possession.py',
        'infer_learned.py','barca4ever.py','games.py','torch_runtime.py','external_idsse.py']
    if args.method=='physics':sources+=['boundary_physics.py','touch_physics.py']
    source_hashes={name:digest(ROOT/name) for name in sources}
    evidence_hashes={name:digest(ROOT/name) for name in evidence}
    pending=target.with_name(target.stem+'.pending.joblib')
    if pending.exists():raise RuntimeError('Preserve pending artifact for inspection')
    joblib.dump(artifact,pending,compress=3)
    restored=joblib.load(pending)
    for key,value in artifact['neural_state'].items():assert np.array_equal(restored['neural_state'][key],value)
    for key in ['tree_model','boundary_model','pair_boundary_model']:
        if key in artifact:assert joblib.hash(restored[key])==joblib.hash(artifact[key])
    assert restored['tree_model'].classes_.tolist()==[0,1,2]
    metadata=dict(kind='refined_contact',version='5',method=args.method,model_sha256=digest(pending),
        parent_model_sha256=parent_meta['model_sha256'],input_mode=artifact['input_mode'],rate=artifact['rate'],
        neural_weight=artifact['neural_weight'],pair_alpha=artifact['pair_alpha'],pair_feature_mode=artifact['pair_feature_mode'],
        architecture=artifact['architecture'],inference_device='cpu',role_mode='none',
        training_matches=SPLITS['train'],licence='CC BY 4.0',attribution=parent_meta['attribution'],
        evidence='Reused external validation, whole-match development comparisons and reused regression; not a new blind benchmark',
        gates={**{'cv_'+k:v for k,v in cross['gates'].items()},**{'reg_'+k:v for k,v in regression['gates'].items()}},
        code_sha256=source_hashes,evidence_sha256=evidence_hashes,promotion_source_sha256=digest(__file__))
    metadata_pending=Path(str(pending)+'.metadata.json')
    metadata_pending.write_text(json.dumps(metadata,indent=2)+'\n')
    pending.replace(target)
    metadata_pending.replace(Path(str(target)+'.metadata.json'))
    print('V5 frozen after all quality gates; neural/model serialization verified; V4.1 preserved.')
