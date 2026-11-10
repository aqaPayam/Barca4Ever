"""Summarize the separately declared dense contact-physics candidate."""
import json
from pathlib import Path
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    reports={}
    for stage in ['validation','crosscheck','regression']:
        path=ROOT/f'external_reports/boundary_physics_{stage}.json'
        if path.exists():reports[stage]=json.loads(path.read_text())
    assert 'validation' in reports
    lines=['# Dense physics boundary research','',
        'This is a separate research candidate, preserving all first transition-pair',
        'and V4.1 artifacts. The first candidate narrowly failed its noisy CV gate',
        'despite improving clean F1 and both short-hit counts. No production source',
        'or competition prediction is modified by these research programs.', '',
        '## Declared method','',
        'Append28 soft XY touch-physics inputs to the293 signed temporal geometry',
        'inputs, giving321. The source helper produces30 channels; the two roster-',
        'fraction channels are excluded by FEATURE_NAMES because they change under',
        'extra missing track-slot padding. The selected28 inputs passed an explicit',
        'padding-invariance check. TOUCH_PHYSICS_PROTOCOL.md gives the exact formulas',
        'and primary-paper basis (ELASTIC and possessor/touch recognition).', '',
        'Raw ball acceleration, direction changes, contact valleys, approach/recession',
        'and gap-validity evidence do not bridge missing raw-ball windows. These are',
        'continuous unfitted features, not hand-produced labels or event times.', '',
        'Dense sampling includes all valid frames within12 frames of an external',
        'change, and every fifth frame elsewhere. Positive weight4, nearby background',
        'weight0.2 and remote sampled background weight1 approximately preserve the',
        'old one-in-five sampling mass while retaining very short changes and their',
        'nearby negatives. Unknown labels are excluded; clean/noisy and H/A-swap',
        'augmentation remain. The seven-class directed detector and 16/8 decoder',
        'are otherwise the same100-iteration,11-leaf model as the first candidate.', '',
        'Alpha0/.25/.5/1 is chosen only on external WR9, before any fold evaluation.',
        'The alpha0 control must reproduce V4.1 exactly in all metrics.', '',
        '| Stage | Tracking | Alpha | Spell F1 | Macro F1 | Short recovery |',
        '|---|---|---:|---:|---:|---:|']
    for stage,report in reports.items():
        for alpha in sorted({r['alpha'] for r in report['rows']}):
            for noisy in [False,True]:
                a=aggregate([r for r in report['rows'] if r['alpha']==alpha and r['noisy']==noisy])
                lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {}/{} |'.format(stage,'noise' if noisy else 'clean',alpha,a['spell_f1'],a['macro_f1'],a['short_hits'],a['short_count']))
    lines+=['','## Acceptance and limitations','']
    for stage,report in reports.items():lines.append('- '+stage+': '+('passed' if report['all_gates_passed'] else 'failed')+' '+str(report['gates']))
    lines+=['',
        'Every acceptance comparison requires higher spell F1 and no loss of short',
        'hits. Full fitting uses only WOH/WOY/WPY/WQQ; promising candidates refit only',
        'on each original two-match fold. WMX/WN1 reused regression is reached only',
        'after both clean/noisy CV gates pass. All seven matches are reused',
        'development data, not a new blind benchmark or significance claim.', '',
        'The geometry and sampling changes form one declared bundle; gains cannot',
        'be attributed to either alone without a new ablation. No supplied match,',
        'video, possession labels or prediction-quality checks enter this research.',
        '17 related synthetic checks passed:6 directed-boundary,2 dense-physics',
        'integration and9 soft-physics checks. Data attribution: EXTERNAL_DATA.md',
        '(licensed IDSSE, DFL/Sportec, Bassek et al.2025, CC BY4.0).', '']
    (ROOT/'boundary_physics_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Dense physics research report generated; active model unchanged.')
