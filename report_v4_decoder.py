"""Report the predefined V4 decoder comparison, including any failed gates."""
import json
from pathlib import Path
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    reports={stage:json.loads((ROOT/f'external_reports/v4_decoder_{stage}.json').read_text()) for stage in ['validation','crosscheck','regression']}
    selected={k:reports['validation']['selected'][k] for k in ['base','scale']}
    for report in reports.values():
        assert all(report['selected'][k]==value for k,value in selected.items())
    gates={}
    rows=[]
    for stage,report in reports.items():
        scenarios=[None] if stage=='validation' else [False,True]
        for noisy in scenarios:
            summaries=[]
            for name in ['baseline','selected']:
                subset=[r for r in report[name]['rows'] if noisy is None or r['noisy']==noisy]
                summaries.append(aggregate(subset))
            old,new=summaries
            key=stage+('_mixed' if noisy is None else '_noise' if noisy else '_clean')
            gates[key+'_spell']=new['spell_f1']>old['spell_f1']
            gates[key+'_short']=new['short_hits']>=old['short_hits']
            rows.append(dict(stage=stage,tracking='mixed' if noisy is None else 'noise' if noisy else 'clean',baseline=old,selected=new))
    summary=dict(status='External development comparison only; decoder-only trial, no refitting',selected=selected,
                 gates=gates,all_gates_passed=all(gates.values()),rows=rows)
    (ROOT/'external_reports/v4_decoder_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# V4 decoder experiment','',
        'Frozen V4 weights and the same external observations were used throughout.',
        'Six settings were declared in V4_DECODER_PROTOCOL.md before results were',
        'inspected: base costs 8/12/16 and boundary scales 4/8. The single setting',
        'selected on WR9 was then compared with V4 on the original whole-match',
        'folds and reused WMX/WN1 regression. V4 controls reproduced exactly.', '',
        f"Selected cost: {selected['base']}/(1+{selected['scale']}*onset); V4 cost: 16/(1+4*onset).", '',
        '| Comparison | Tracking | V4 spell F1 | Trial spell F1 | V4 short | Trial short |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        a,b=row['baseline'],row['selected']
        lines.append('| {} | {} | {:.4f} | {:.4f} | {}/{} | {}/{} |'.format(row['stage'],row['tracking'],a['spell_f1'],b['spell_f1'],a['short_hits'],a['short_count'],b['short_hits'],b['short_count']))
    lines+=['', 'Acceptance gates: '+('all passed.' if all(gates.values()) else 'failed; preserve V4.'), '',
        'These are reused external development comparisons, not fresh blind tests.',
        'Small gains do not establish statistical significance or professional-system',
        'superiority. Clean/noisy rows share labels. No supplied competition match,',
        'video, possession labels or output-quality checks were used. Frozen V4',
        'artifacts and predictions are preserved. Per-half results and gate details',
        'are in external_reports/v4_decoder_*.json.', '',
        '## Diagnostic motivation and remaining work', '',
        'On frozen V4 external validation, 11 noisy short spells were matched by raw',
        'labels but lost after decoding. Among misses, 20 spells had a correct',
        'majority from one component which was lost by probability averaging.',
        'These are overlapping descriptive symptoms, not guaranteed recoverable',
        'gains or an oracle performance target. The diagnostic is reproducible with',
        'diagnose_v4_short.py and its source/model hashes are retained.', '',
        'Further hypotheses are reliability-dependent fusion learned from strictly',
        'out-of-fold predictions, corruption of player identities during external',
        'training, and an additional 25-Hz contact/boundary path. Each needs its own',
        'validation and whole-match comparison. Broader licensed external matches',
        'and a genuinely unseen benchmark are needed to assess transfer reliably.', '']
    (ROOT/'V4_DECODER_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary),flush=True)
