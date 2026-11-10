"""Freeze the selected external-only candidate; no access to competition inputs."""
import hashlib
import json
from pathlib import Path
import shutil
from report_v2 import aggregate

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    reports = ROOT / 'external_reports/research_v3'
    val = json.loads((reports / 'learned_boundary.json').read_text())['selected']
    oldval = json.loads((ROOT / 'external_reports/v2_validation.json').read_text())['selected']
    gates = {'validation_spell_f1': val['spell_f1'] > oldval['spell_f1']}
    for file in ['boundary_crosscheck', 'boundary_regression']:
        rows = json.loads((reports / (file + '.json')).read_text())['rows']
        for noisy in [False, True]:
            a, b = [aggregate([r for r in rows if r['model'] == model and r['noisy'] == noisy])
                    for model in ['v2', 'learned_boundary']]
            for metric in ['spell_f1', 'short_recall']:
                gates[f'{file}_{noisy}_{metric}'] = b[metric] > a[metric]
    if not all(gates.values()):
        raise RuntimeError('External acceptance checks failed: ' + str(gates))
    target = ROOT / 'models/external_v3.joblib'
    source = ROOT / 'models/research_v3/learned_boundary.joblib'
    if target.exists() and digest(target) != digest(source):
        raise RuntimeError('Existing frozen V3 differs; preserve it before further experiments')
    shutil.copyfile(source, target)
    code = ['possession_v2.py', 'learned_possession.py', 'infer_learned.py',
            'infer_boundary.py', 'temporal_decoders.py', 'research_data.py',
            'research_trees.py', 'develop_v2.py', 'external_idsse.py']
    metadata = dict(model_sha256=digest(target), kind='adaptive_boundary', feature_version=2,
        penalty=val['penalty'], boundary_scale=val['boundary_scale'], gates=gates,
        training_matches=['J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ'],
        validation_match='J03WR9', licence='CC BY 4.0',
        attribution='DFL/Sportec Solutions; IDSSE, Bassek et al. (2025); DOI 10.6084/m9.figshare.28196177',
        evidence='Development validation, whole-match cross-validation, reused-test regression; not fresh blind tests. Noisy CV overall gain is negligible.',
        acceptance='Post-development acceptance checks, not a preregistered blind-test protocol',
        code_sha256={name: digest(ROOT / name) for name in code})
    Path(str(target) + '.metadata.json').write_text(json.dumps(metadata, indent=2))
    print('V3 frozen using external evidence only. V1 and V2 preserved unchanged.')
