"""Human-readable audit trail for transition-pair boundary research."""
import json
from pathlib import Path
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    reports={}
    for stage in ['validation','crosscheck','regression']:
        path=ROOT/f'external_reports/boundary_refinement_{stage}.json'
        if path.exists():reports[stage]=json.loads(path.read_text())
    assert 'validation' in reports
    validation=reports['validation'];chosen=validation['selected']['alpha']
    lines=['# Transition-pair boundary refinement','',
        'Research-only candidate; V4.1 remains frozen until all acceptance and',
        'production checks are complete. No competition inputs or predictions',
        'are read by the research program.', '',
        '## Method','',
        'V4.1 predicts only the incoming state at a potential change. This candidate',
        'predicts six ordered source/destination transitions plus background. A',
        'separate score controls H-to-A versus D-to-A, for example. Signed before/',
        'after geometry at +/-2,6,12 frames supplements the original 59 inputs,',
        'for293 total. The original control used absolute differences at +/-3.', '',
        'The classifier is a100-iteration,11-leaf histogram gradient booster with',
        'rate0.08,L2=5,minleaf80,seed20261116. Changes within two frames provide',
        'targets; nearest change wins, with earlier changes winning ties. Positive',
        'frames have weight20. Training samples every fifth valid frame, using',
        'clean/noisy observations and H/A swap augmentation. No classifier or',
        'network probability output is a feature, so training the boundary model',
        'does not rely on in-sample expert predictions.', '',
        'Positive-label weighting means the directed outputs are cost-control scores,',
        'not calibrated probabilities of a real-world transition.', '',
        'A directed-score Viterbi decoder retains base16/scale8. New scores blend',
        'with frozen V4.1 destination scores at alpha0,.25,.5,1, selected on WR9.',
        'Alpha0 is mathematically and empirically the exact V4.1 control.', '',
        f'WR9 selected alpha: {chosen}.', '',
        '## External comparison','',
        '| Stage | Tracking | Alpha | Spell F1 | Macro F1 | Short recovery |',
        '|---|---|---:|---:|---:|---:|']
    for stage,report in reports.items():
        alphas=sorted({r['alpha'] for r in report['rows']})
        for alpha in alphas:
            for noisy in [False,True]:
                selected=[r for r in report['rows'] if r['alpha']==alpha and r['noisy']==noisy]
                a=aggregate(selected)
                lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {}/{} |'.format(stage,'noise' if noisy else 'clean',alpha,a['spell_f1'],a['macro_f1'],a['short_hits'],a['short_count']))
    lines+=['','## Acceptance and limits','']
    for stage,report in reports.items():lines.append('- '+stage+': '+('passed' if report['all_gates_passed'] else 'failed')+' '+str(report['gates']))
    lines+=['',
        'Training uses only the four licensed external IDSSE training games; WR9',
        'selects alpha. Promising designs are fitted again on each original two-match',
        'fold and compared on the other two matches. WMX/WN1 reused regression is',
        'only reached after both clean/noisy CV gates pass. Every comparison demands',
        'higher spell F1 and no short-hit loss, and V4.1 controls reproduce exactly.', '',
        'All seven games have appeared in development. This is not a new blind',
        'benchmark, statistical-significance claim or professional-system ranking.',
        'The method is an original refinement inspired by boundary supervision',
        '(Ishikawa et al., WACV2021) and possession/event timing research; it is not',
        'their ASRF architecture. Sources and fixed choices are recorded in',
        'boundary_refinement_protocol.md. IDSSE attribution: EXTERNAL_DATA.md.', '',
        'Six synthetic checks passed: target direction and unknown handling, H/A',
        'mapping and nearest-boundary ties, signed temporal direction, exact V4.1',
        'control recovery, ordered-edge probability routing, and brute-force',
        'verification of the decoder objective.', '']
    (ROOT/'boundary_refinement_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Transition-pair research report written; active model unchanged.')
