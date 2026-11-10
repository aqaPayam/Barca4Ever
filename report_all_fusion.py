"""Consolidate the declared adaptive-combination cycle, retaining all failures."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    families=[('Linear','smart_fusion'),('Conservative','conservative_fusion'),
              ('Nonlinear','nonlinear_fusion'),('Spell-score fitted','score_fusion')]
    baseline=None;rows=[];selected=[];reports={}
    for label,name in families:
        report=json.loads((ROOT/f'external_reports/{name}_crosscheck.json').read_text())
        reports[name]=report
        if baseline is None:baseline=report['summary']['v4.1']
        else:assert baseline==report['summary']['v4.1']
        for candidate,scores in report['summary'].items():
            if candidate=='v4.1':continue
            rows.append(dict(family=label,candidate=candidate,scores=scores,checks=report['gates'][candidate]))
        if report['selected']:selected.append((name,report['selected']))
    assert len(selected)<=1,'Sequential experiment protocol expected only one confirming candidate'
    regression=None;accepted=False
    if selected:
        family,candidate=selected[0]
        regression=json.loads((ROOT/f'external_reports/{family}_regression.json').read_text())
        accepted=regression['selected']==candidate
    baseline_path=ROOT/'models/external_v4_1.joblib'
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert all(r['baseline_sha256']==digest(baseline_path) for r in reports.values())
    protocols=['SMART_FUSION_PROTOCOL.md','SMART_FUSION_CONSERVATIVE_PROTOCOL.md','NONLINEAR_FUSION_PROTOCOL.md','SCORE_FUSION_PROTOCOL.md']
    summary=dict(status='Reused external development comparisons; WR9 fits gates, whole-match CV screens, only one candidate can confirm on reused regression',
        baseline_model_sha256=digest(baseline_path),baseline=baseline,candidates=rows,
        selected_on_cv=selected,external_acceptance_passed=accepted,regression=regression,
        protocol_sha256={p:digest(ROOT/p) for p in protocols})
    (ROOT/'external_reports/all_smart_fusion_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Learned combinations compared with V4.1','',
        'This cycle implemented and tested four families of time-varying reliability',
        'combination. Every original model weight and the 16/8 decoder stayed fixed.',
        'The component predictions, original match splits and corruption seeds match',
        'the V4.1 benchmark. Baseline spell scores reproduced exactly throughout.', '',
        '## Fitting and evaluation roles','',
        'WR9 changes from validation to gate-fitting/calibration data. Each gate sees',
        'WR9 predictions from base weights that were not fitted on WR9. However, WR9',
        'was previously used to select architecture/checkpoints; it is reused',
        'development data, not strict new blind cross-fitting. No WR9 fitting score',
        'is used to claim generalization. The same seven licensed IDSSE games are used.', '',
        'Each whole-match fold has its own gate. The scaler, base trees, neural',
        'initializer, neural adaptation and gate fitting exclude the evaluated fold',
        'matches. CV screens candidates; only a qualifying single choice goes to',
        'WMX/WN1 reused regression. These comparisons are development evidence, not',
        'nested blind evaluation, statistical significance or competition rankings.', '',
        '## Whole-match comparison','',
        '| Family | Candidate | Clean spell F1 | Noisy spell F1 | Clean short /463 | Noisy short /463 |',
        '|---|---|---:|---:|---:|---:|']
    def table_row(family,name,s):
        a,b=s['False'],s['True']
        return '| {} | {} | {:.4f} | {:.4f} | {} | {} |'.format(family,name,a['spell_f1'],b['spell_f1'],a['short_hits'],b['short_hits'])
    lines.append(table_row('Baseline','V4.1',baseline))
    for row in rows:lines.append(table_row(row['family'],row['candidate'],row['scores']))
    lines+=['','Selection rule: improve spell F1 and avoid a short-hit decline in BOTH',
        'clean/noisy cases. Extra short recovery alone does not qualify. Short',
        'recovery counts correctly matched team possessions under two seconds at',
        'IoU0.5; spell F1 is the exact organizer mean over IoU0.3 through0.7.', '',
        'The spell-score fitted algorithm internally selects two bounded parameters',
        'from 25 settings on each fold WR9 calibration match. Those are fitting',
        'trials, not extra independent test results. All calibration choices are',
        'retained in models/smart_fusion/score_fold*.json.', '']
    if accepted:
        lines+=['One candidate passed external acceptance. Production checks are recorded',
            'separately before the active runner or prediction file can change.', '']
    else:
        lines+=['No candidate passed the complete acceptance rule. V4.1 remains the active',
            'model; its artifact, runner and local prediction file are preserved.',
            'No failed candidate was evaluated to search the regression matches for',
            'a favorable replacement. Research code, fitted gates and negative results',
            'are retained for reproducibility, separate from the active model.', '']
    if regression:
        lines+=['## Single-candidate reused regression','',
            '| Model | Clean spell F1 | Noisy spell F1 | Clean short /362 | Noisy short /362 |',
            '|---|---:|---:|---:|---:|']
        for name,s in regression['summary'].items():
            a,b=s['False'],s['True']
            lines.append('| {} | {:.4f} | {:.4f} | {} | {} |'.format(name,a['spell_f1'],b['spell_f1'],a['short_hits'],b['short_hits']))
        lines+=['']
    lines+=['## Implementations and checks','',
        'The first linear gate used bounded likelihood fitting; factorized used',
        'separate dead/conditional-team gates. Conservative variants restricted',
        'their strength or activation. The nonlinear gate predicted relative',
        'true-label probability advantage. The final small reliability rule fitted',
        'spell scores on calibration only. Protocols were recorded before each family',
        'was evaluated; see the four *_PROTOCOL.md files and per-half JSON reports.', '',
        'Six synthetic checks passed: finite-difference gradient validation, learning',
        'a reliability signal, probability sums and bounds, H/A exchange invariance',
        'including ties, exact zero-parameter baseline and model serialization.',
        'No competition tracking, video, labels or output-quality checks were used.',
        'External-data attribution and CC BY4.0 licence: EXTERNAL_DATA.md.', '']
    (ROOT/'SMART_FUSION_ALL_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Adaptive-fusion cycle documented:',len(rows),'CV variants; acceptance:',accepted)
