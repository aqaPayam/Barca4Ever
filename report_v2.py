"""Summarize external-only v2 evidence, with explicit reuse caveats."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def aggregate(rows):
    buckets=[r['recall_by_duration']['under_2s'] for r in rows]
    count=sum(b['count'] for b in buckets)
    hits=sum(round(b['count']*b['recall_iou50']) for b in buckets if b['count'])
    return dict(spell_f1=float(np.mean([r['spell_f1'] for r in rows])),
                macro_f1=float(np.mean([r['macro_f1'] for r in rows])),
                short_hits=hits,short_count=count,short_recall=hits/count if count else 0)


if __name__=='__main__':
    reports=ROOT/'external_reports'
    validation=json.loads((reports/'v2_validation.json').read_text())
    oldval=json.loads((reports/'v1_short_spell_validation.json').read_text())
    cross=json.loads((reports/'v2_crosscheck.json').read_text())
    regression=json.loads((reports/'v2_regression.json').read_text())
    summary=[]
    for dataset,rows in [('whole_match_cross_validation',cross['rows']),('reused_test_regression',regression['rows'])]:
        for noisy in [False,True]:
            for name in ['v1','v2']:
                group=[r for r in rows if r['noisy']==noisy and r['model']==name]
                summary.append(dict(dataset=dataset,noisy=noisy,model=name,**aggregate(group)))
    for name,rows in [('v1',oldval),('v2',validation['selected']['rows'])]:
        summary.append(dict(dataset='validation_selection',model=name,**aggregate(rows)))
    (reports/'v2_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Short-possession model v2: external results','',
        'The competition match was not used for fitting, tuning, selection or quality scoring.',
        'V1 and its submission remain available as a fallback.','',
        '## Changes','',
        '- Add 17 features (59 total): short-gap ball interpolation, nearest-team geometry',
        '  after interpolation, relative player/ball motion, directional alignment and',
        '  near-contact context at 0.12 seconds. Interpolation never extrapolates beyond',
        '  observations and leaves gaps longer than two seconds missing.',
        '- Weight externally labelled team spells by clipped square-root inverse duration:',
        '  clip(sqrt(250 / duration_frames), 0.5, 4). Dead-ball frames keep weight 1.',
        '- Fit 200 boosting iterations with 23 leaves and select a temporal switch',
        '  penalty from 2/4/8/16/32 using only J03WR9. Selected penalty: 8.','',
        '## Evidence','',
        'Whole-match cross-validation retrains BOTH models on identical two-match folds',
        'from the original four training games. Each evaluated match is absent from',
        'both fitted models in its fold. These are development data, not a new blind test.',
        'The WMX/WN1 comparison reuses previously inspected test games and is explicitly',
        'a regression check. No parameter changes followed those regression results.','',
        'Short-spell recall counts true possessions under two seconds recovered at IoU 0.5.',
        'Clean and noisy scenarios reuse the same underlying labels; their counts are',
        'not independent observations. Score means weight halves equally.','',
        '| Evaluation | Noise | Model | Spell F1 | Macro F1 | Short-spell recall |',
        '|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {}/{} ({:.1%}) |'.format(
            r['dataset'],r.get('noisy','mixed'),r['model'],r['spell_f1'],r['macro_f1'],
            r['short_hits'],r['short_count'],r['short_recall']))
    lines+=['','## Limits','',
        'Data remains limited and provider-specific. Synthetic noise does not reproduce',
        'all challenge tracking errors. External improvements do not establish a',
        'competition score or rank. A genuinely new blind test requires additional',
        'licensed matches not already used in this project.',
        'Full data attribution is in EXTERNAL_DATA.md. Exact results are retained in',
        'external_reports/v2_validation.json, v2_crosscheck.json and v2_regression.json.','']
    (ROOT/'V2_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary),flush=True)
