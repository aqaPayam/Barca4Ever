"""Summarize predefined frozen-network symmetry averaging without promotion."""
import json
from pathlib import Path
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    validation=json.loads((ROOT/'external_reports/tta_validation.json').read_text())
    lines=['# Frozen-network symmetry averaging','',
        'Three candidates were declared in TTA_PROTOCOL.md before scoring: average',
        'the original NN joint probabilities with restored H/A-swapped predictions,',
        'with a180-degree pitch reflection, or with both. Tree mixing stays0.5 and',
        'the state-boundary decoder stays16/8. Every model weight is unchanged.', '',
        'Transform original11-channel inputs, preserve missing sentinels and roles,',
        'then derive relative17-channel features. Restore H/A output columns before',
        'averaging full25Hz joint posteriors. All new forward passes are CPU-only',
        'with2threads. Existing original-path posteriors may be reused bit-exactly',
        'from the verified external component caches; provenance is recorded in',
        'external_reports/tta_original_reuse_audit.json.', '',
        '## External WR9 selection','',
        '| Candidate | Mixed spell F1 | Short hits /188 | Clean spell F1 | Noisy spell F1 |',
        '|---|---:|---:|---:|---:|']
    names=['v4.1','swap','reflection','both']
    for name in names:
        combined=aggregate([r for r in validation['rows'] if r['model']==name])
        clean=validation['summary'][name]['False']['spell_f1'];noise=validation['summary'][name]['True']['spell_f1']
        lines.append('| {} | {:.4f} | {} | {:.4f} | {:.4f} |'.format(name,combined['spell_f1'],combined['short_hits'],clean,noise))
    lines+=['',f"Selected: {validation['selected'] or 'none'}.",
        'Selection requires strictly higher mixed spell F1 and no loss of short hits.',
        'Only that selected candidate proceeds; there is no fallback after CV.', '']
    accepted=False;stages=[]
    for stage in ['crosscheck','regression']:
        path=ROOT/f'external_reports/tta_{stage}.json'
        if not path.exists():continue
        report=json.loads(path.read_text());stages.append(stage)
        lines+=['## '+('Whole-match development comparison' if stage=='crosscheck' else 'Reused-test regression'),'',
            '| Model | Clean spell F1 | Noisy spell F1 | Clean short | Noisy short |',
            '|---|---:|---:|---:|---:|']
        for name,summary in report['summary'].items():
            a,b=summary['False'],summary['True']
            lines.append('| {} | {:.4f} | {:.4f} | {}/{} | {}/{} |'.format(name,a['spell_f1'],b['spell_f1'],a['short_hits'],a['short_count'],b['short_hits'],b['short_count']))
        lines+=['','Stage acceptance: '+('passed.' if report['accepted'] else 'failed; preserve V4.1.'),'']
        if stage=='regression':accepted=report['accepted']
    lines+=['## Reproducibility and limits','',
        'The original whole-match folds retain their own role-free NN and tree/onset',
        'models, with evaluated games excluded from all fitting. Controls must match',
        'V4.1 exactly in spell F1, macro F1 and duration recalls. These are the same',
        'seven licensed external IDSSE games already used in development, not fresh',
        'blind tests, evidence of statistical significance or a competition ranking.',
        'No supplied competition input, video, label or prediction quality is used.', '',
        'Three synthetic tests pass: H/A transform and output-label inverses,',
        'reflection inverse with actual sign changes and missing padding preserved,',
        'and normalized joint-probability averaging. Original sources and artifacts',
        'remain untouched. Passing external comparisons is distinct from production',
        'validation and promotion, which are not performed by these scripts.', '',
        '```text','python -m unittest tests.test_tta_features -v',
        'python tta_research.py --stage validation','python tta_research.py --stage crosscheck',
        'python tta_research.py --stage regression','python tta_report.py','```','',
        'Each later stage requires the prior acceptance gate. Existing evidence is',
        'preserved; reproduce in a fresh experiment directory. Dependencies, base',
        'fitting and external data attribution: V4_MODEL_GUIDE.md, V4_1_MODEL_GUIDE.md',
        'and EXTERNAL_DATA.md. The data licence is CC BY4.0.', '']
    (ROOT/'TTA_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('TTA report generated; stages available:',stages,'external confirmation passed:',accepted)
