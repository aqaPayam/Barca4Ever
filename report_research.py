"""Summarize the complete external-only architecture search, including failures."""
import json
from pathlib import Path
from report_v2 import aggregate

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'external_reports/research_v3'


def summarize():
    rows = []
    v2 = json.loads((ROOT / 'external_reports/v2_validation.json').read_text())
    rows.append(dict(evaluation='validation_selection', model='v2',
                     **aggregate(v2['selected']['rows'])))
    for name in ['hierarchical', 'learned_boundary', 'tcn', 'fusion']:
        report = json.loads((REPORTS / (name + '.json')).read_text())
        rows.append(dict(evaluation='validation_selection', model=name,
                         **aggregate(report['selected']['rows'])))
    for file, evaluation in [('boundary_crosscheck', 'cross_validation'),
                             ('fusion_crosscheck', 'cross_validation'),
                             ('boundary_regression', 'reused_test_regression')]:
        report = json.loads((REPORTS / (file + '.json')).read_text())
        for noisy in [False, True]:
            for model in sorted({r['model'] for r in report['rows']}):
                if file == 'fusion_crosscheck' and model == 'v2':
                    continue  # The same control results are already included.
                group = [r for r in report['rows']
                         if r['model'] == model and r['noisy'] == noisy]
                rows.append(dict(evaluation=evaluation, model=model, noisy=noisy,
                                 **aggregate(group)))
    return rows


if __name__ == '__main__':
    summary = summarize()
    (REPORTS / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Model research results: boundary-aware v3', '',
        'The selected model is the strongest tested candidate in this search. This',
        'does not establish a globally optimal model or a competition score.', '',
        '## Research and implemented candidates', '',
        'We reviewed exact-task football possession TCN research and temporal action',
        'segmentation research. Primary sources and design decisions are recorded in',
        '[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). Implementations are original;',
        'we did not copy paper code or obtain their confidential training data/weights.', '',
        'Twenty-six validation configurations were evaluated in addition to the V2',
        'control: five hierarchical dead/alive + team penalties; six boosted boundary',
        'decoder settings; nine compact TCN checkpoint/penalty pairs; four TCN/tree',
        'blends and two exploratory TCN-boundary decoder settings.', '',
        'The neural model has 59 inputs, 32 channels, five dilated residual blocks',
        '(dilations 1/2/4/8/16), no temporal pooling, and possession/boundary heads.',
        'It was trained from scratch on external data, with short-spell weighting.',
        'Neither the TCN alone nor its blend is used in production. The blend gained',
        'on validation but reduced short-spell recall in cross-validation.', '',
        '## Selected design', '',
        'Input: player/ball XY tracking, team metadata and missingness. The unchanged',
        'V2 classifier estimates home/away/dead from 59 geometry/motion features.',
        'A second boosted classifier estimates transition evidence from 177 features:',
        'the 59 inputs and absolute differences from three frames before/after.',
        'External label changes widened by three frames provide its supervision.',
        'It has 140 iterations, 15 leaves, learning rate 0.08, L2 3 and minimum leaf',
        'size 60; positive boundary frames carry weight 10. Its output b is a score,',
        'not a calibrated probability because training uses unequal class weights.', '',
        'The decoder maximizes summed log class probabilities minus switch costs',
        'C_t = 16 / (1 + 4 b_t). This permits faster changes where boundary evidence',
        'is strong and stronger continuity where it is weak. Output remains one',
        'H/A/D/X label per frame. Unknown long tracking outages use the prior X rule.', '',
        '## External results', '',
        'Spell F1 is the organizer metric averaged equally over evaluated halves.',
        'Short recall counts possessions under two seconds matched at IoU 0.5.',
        'Clean/noisy variants share labels and are not independent samples.', '',
        '| Evaluation | Tracking | Model | Spell F1 | Macro F1 | Short recall |',
        '|---|---|---|---:|---:|---:|']
    for r in summary:
        scenario = ('noise' if r['noisy'] else 'clean') if 'noisy' in r else 'mixed'
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {}/{} ({:.1%}) |'.format(
            r['evaluation'], scenario, r['model'], r['spell_f1'], r['macro_f1'],
            r['short_hits'], r['short_count'], r['short_recall']))
    lines += ['', '## Evidence and limits', '',
        'Training/selection: four IDSSE matches, validation J03WR9, CC BY 4.0.',
        'Cross-validation: fit WOH/WOY and evaluate WPY/WQQ, then reverse. Models',
        'were fitted again within each fold; both halves stay together. Boundary',
        'parameters and neural training length were fixed before these comparisons.',
        'The fitted model never saw its fold evaluation match. Architecture selection',
        'still uses development evidence; these are not a new independent blind test.', '',
        'The noisy cross-validation spell F1 change (0.4170 to 0.4178) is essentially',
        'flat and should not be treated as a reliable overall gain. Some individual',
        'halves declined. The clean gain and short-spell recall gains are clearer.',
        'The WMX/WN1 results are regression on previously inspected external matches;',
        'no parameter changes followed that check. No significance or ranking claim',
        'is made from seven previously used games or synthetic noise.', '',
        'All competition assets remain inference-only. We did not score predictions',
        'against the competition ground truth, compare them against its video, or',
        'fit/tune on the supplied match. Its CSV is checked only for file validity.',
        'New licensed matches are needed for a genuinely fresh blind assessment.', '',
        '## Reproduction', '',
        'V3 inference uses requirements-model.txt only, without PyTorch. For research',
        'also install requirements-research.txt. See V3_MODEL_GUIDE.md for commands.',
        'All trials and per-half results, including negative results, are retained',
        'in external_reports/research_v3/. Data attribution is in EXTERNAL_DATA.md.', '']
    (ROOT / 'MODEL_RESEARCH_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('External research summary written; no competition data accessed.')
