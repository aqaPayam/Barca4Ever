"""Summarize like-for-like external comparisons without overstating replication."""
import hashlib
import json
from pathlib import Path
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


def row_summary(rows):
    result=aggregate(rows)
    values=[r['global_accuracy'] for r in rows if 'global_accuracy' in r]
    result['global_accuracy']=sum(values)/len(values) if values else None
    return result


if __name__=='__main__':
    summary=[]
    old=json.loads((ROOT/'external_reports/research_v3/learned_boundary.json').read_text())['selected']
    summary.append(dict(evaluation='validation',model='v3',tracking='mixed',**row_summary(old['rows'])))
    for recipe in ['paper','adapted','relative','rolefree']:
        path=ROOT/f'models/paper2023/{recipe}.validation.json'
        if not path.exists():continue
        report=json.loads(path.read_text())
        for name,selected in [(recipe,report['selected'])]+([('paper_raw',report['trials'][0])] if recipe=='paper' else []):
            summary.append(dict(evaluation='validation',model=name,tracking='mixed',**row_summary(selected['rows'])))
    optional=[('models/paper2023/fusion.validation.json','fusion'),
              ('models/paper2023/fusion_rolefree.validation.json','fusion_rolefree'),
              ('external_reports/state_boundary_validation.json','state_boundary'),
              ('external_reports/blend_boundary_validation.json','boundary_blend')]
    for file,name in optional:
        path=ROOT/file
        if path.exists():
            selected=json.loads(path.read_text())['selected']
            summary.append(dict(evaluation='validation',model=name,tracking='mixed',**row_summary(selected['rows'])))
    for file,label in [('paper2023_crosscheck.json','cross-validation'),
                       ('paper2023_rolefree_crosscheck.json','cross-validation'),
                       ('state_boundary_crosscheck.json','cross-validation')]:
        path=ROOT/'external_reports'/file
        if not path.exists():continue
        rows=json.loads(path.read_text())['rows']
        for model in sorted({r['model'] for r in rows}):
            if file.startswith('state') and model=='v3':continue
            if file.startswith('paper2023_rolefree') and model=='v3':continue
            for noisy in [False,True]:
                subset=[r for r in rows if r['model']==model and r['noisy']==noisy]
                name='fusion_rolefree' if file.startswith('paper2023_rolefree') and model=='fusion' else model
                summary.append(dict(evaluation=label,model=name,tracking='noise' if noisy else 'clean',**row_summary(subset)))
    for recipe in ['paper','adapted','relative','rolefree','fusion','fusion_rolefree']:
        path=ROOT/f'external_reports/paper2023_regression_{recipe}.json'
        if not path.exists():continue
        rows=json.loads(path.read_text())['rows']
        for model in sorted({r['model'] for r in rows}):
            if recipe!='paper' and model=='v3':continue
            for noisy in [False,True]:
                subset=[r for r in rows if r['model']==model and r['noisy']==noisy]
                name=recipe if recipe.startswith('fusion') else model
                summary.append(dict(evaluation='reused-test regression',model=name,
                    tracking='noise' if noisy else 'clean',**row_summary(subset)))
    state_path=ROOT/'external_reports/state_boundary_regression.json'
    if state_path.exists():
        rows=json.loads(state_path.read_text())['rows']
        for noisy in [False,True]:
            subset=[r for r in rows if r['model']=='state_boundary' and r['noisy']==noisy]
            summary.append(dict(evaluation='reused-test regression',model='state_boundary',
                 tracking='noise' if noisy else 'clean',**row_summary(subset)))
    files=['paper2023_model.py','paper2023_data.py','train_paper2023.py','crosscheck_paper2023.py',
           'PAPER_2023_PROTOCOL.md','torch_runtime.py','research_state_boundaries.py']
    evidence=dict(status='External development comparisons only; not a fresh blind test or original-author reproduction',
        summary=summary,source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (ROOT/'external_reports/paper2023_summary.json').write_text(json.dumps(evidence,indent=2))
    lines=['# Paper-based possession model: implementation and external comparison','',
        'Independent PyTorch implementation of Borghesi et al. (2023), ESWA 223,',
        '119780: https://doi.org/10.1016/j.eswa.2023.119780. Original trained weights,',
        'code and confidential data were unavailable. This tests a retrained',
        'paper-based adaptation; it does not establish the authors original score.', '',
        '## Implementation fidelity', '',
        'The strongest published architecture (two branches, separate ball-query',
        'self-attention, shared per-object trajectory TCN) is implemented directly,',
        'rather than the earlier hand-feature TCN. It uses 64 timesteps at 16 Hz,',
        'target offset 47, 36 object slots, 11 input channels, causal dilated',
        'convolution blocks, batch normalization, dropout, and ELU 64/32-unit FFNs.',
        'Home/away loss is zero on externally labelled dead frames. See',
        'PAPER_2023_PROTOCOL.md for every unspecified setting and adaptation.', '',
        'IDSSE lacks referee trajectories: a missing referee slot is retained.',
        'Goalkeepers use external metadata. Missing participants are padded.',
        'Predictions are interpolated back to all 25-Hz frames before scoring.', '',
        'The paper recipe keeps clean training, Adam 1e-5 and decay 0.7 per epoch,',
        '100,000 windows and validation-accuracy checkpoint selection. The adapted',
        'recipe changes optimization to Adam 0.001/cosine, adds simulated noise,',
        'team swaps and short-spell weights, and selects by validation spell F1.',
        'The relative recipe also adds six ball-relative contact/motion channels,',
        'scaled velocity inputs, short-gap interpolation, a target-frame geometry',
        'MLP skip, distance-biased attention and 25% uniform-spell training references.',
        'These are bundled original extensions, not components of the paper.', '',
        'The rolefree recipe initializes from the contact model and fine-tunes two',
        'epochs with keeper-role channels zero. This matches the available tactical',
        'input and avoids guessed roles. Each fold is adapted only from its own',
        'eight-epoch checkpoint and only on its own fitting matches. The final',
        'role-free fusion is selected and checked separately from the role-based',
        'research model. No supplied metadata contents or possession observations',
        'were displayed in the local input compatibility check.', '',
        'The published architecture does not prescribe our Viterbi decoder.',
        'paper_raw is its original hierarchical threshold prediction; paper',
        'adds a decoder chosen on external validation. State-boundary is an',
        'independent tree-based extension of V3, not part of the paper.', '',
        '## Results', '',
        'Means weight evaluated halves equally. Short recall counts possessions',
        'under two seconds recovered at IoU 0.5. Clean/noisy rows share labels.', '',
        '| Evaluation | Tracking | Model | Spell F1 | Macro F1 | Short recovery |',
        '|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {}/{} ({:.1%}) |'.format(
            r['evaluation'],r['tracking'],r['model'],r['spell_f1'],r['macro_f1'],
            r['short_hits'],r['short_count'],r['short_recall']))
    lines+=['','## Interpretation limits','',
        'All seven IDSSE matches have appeared in development. Whole-match folds',
        'exclude their evaluated matches from fitting, but architecture acceptance',
        'uses development evidence. This is not nested validation or a new blind',
        'test. The paper used a different private dataset and accuracy metric.',
        'Beating this implementation here cannot establish superiority over the',
        'original trained system, current professional products or all models.', '',
        'No competition video, labels or tracking were used in these experiments.',
        'Competition predictions remain local and cannot establish a quality score.', '',
        '## Reproduction','',
        'Install requirements-paper2023.txt for GPU training, or use the CPU',
        'requirements-contact.txt with the same torch version. The final model',
        'and fold-specific training commands are in V4_MODEL_GUIDE.md. Run the',
        'relative whole-match crosscheck before the role-free adaptation so each',
        'fold has its own external-only initializer. Reproduce in a fresh experiment',
        'directory; existing artifact/report guards preserve original evidence.',
        'Data licence and attribution: EXTERNAL_DATA.md (IDSSE, CC BY 4.0).', '',
        'Synthetic tests cover permutation invariance, loss masking, missingness,',
        'target alignment, direct-window versus dense predictions and exact',
        'destination-cost decoding. Training curves, seeds and artifact hashes',
        'are retained with the models. Source hashes are in the JSON summary.', '']
    (ROOT/'PAPER_2023_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Paper-based model results documented; external data only.')
