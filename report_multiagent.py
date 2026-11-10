"""Consolidate paper-driven team experiments and verify the release choice."""
import hashlib
import json
from pathlib import Path
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


def read(name):return json.loads((ROOT/name).read_text())


if __name__=='__main__':
    baseline=read('external_reports/v4_decoder_validation.json')['selected']
    neural=read('external_reports/hz25_selection.json')
    pair=read('external_reports/boundary_refinement_validation.json')
    physics=read('external_reports/boundary_physics_validation.json')
    composite=read('external_reports/combined_refinement_validation.json')
    tta=read('external_reports/tta_validation.json')
    validation=[dict(name='V4.1',spell_f1=baseline['spell_f1'],short_hits=baseline['short_hits'])]
    for rate in [16,25]:
        trials=read(f'external_reports/hz25_validation_{rate}.json')['trials']
        best=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1']))
        validation.append(dict(name=f'Observability {rate}Hz',spell_f1=best['spell_f1'],short_hits=best['short_hits']))
    for name,report in [('Directed boundary',pair),('Physics+dense boundary',physics),('Composite',composite)]:
        selected=report['selected']
        validation.append(dict(name=name,spell_f1=selected['spell_f1'],short_hits=selected['short_hits']))
    selected_tta=[r for r in tta['rows'] if r['model']==tta['selected']]
    validation.append(dict(name='Symmetry averaging',**aggregate(selected_tta)))
    cv=[];reg=[];eligible=[]
    neural_cv=read('external_reports/hz25_crosscheck.json')
    pair_cv=read('external_reports/boundary_refinement_crosscheck.json')
    physics_cv=read('external_reports/boundary_physics_crosscheck.json')
    composite_cv=read('external_reports/combined_refinement_crosscheck.json')
    tta_cv=read('external_reports/tta_crosscheck.json')
    for label,report,kind in [('Observability16',neural_cv,'model'),('Directed boundary',pair_cv,'alpha'),
                              ('Physics+dense boundary',physics_cv,'alpha'),('Composite',composite_cv,'model'),('Symmetry averaging',tta_cv,'model')]:
        for noisy in [False,True]:
            if kind=='alpha':
                alpha=report['selected']['alpha']
                subset=[r for r in report['rows'] if r['alpha']==alpha and r['noisy']==noisy]
            else:
                name='both' if label=='Symmetry averaging' else 'candidate'
                subset=[r for r in report['rows'] if r['model']==name and r['noisy']==noisy]
            cv.append(dict(name=label,noisy=noisy,**aggregate(subset)))
    for label,prefix,val in [('Physics+dense boundary','boundary_physics',physics['selected']),('Composite','combined_refinement',composite['selected'])]:
        report=read(f'external_reports/{prefix}_regression.json')
        accepted=report.get('accepted',report.get('all_gates_passed',False))
        cross=physics_cv if prefix=='boundary_physics' else composite_cv
        accepted=accepted and cross.get('accepted',cross.get('all_gates_passed',False))
        if accepted:eligible.append(dict(name=label,validation=val,prefix=prefix))
        for noisy in [False,True]:
            subset=[r for r in report['rows'] if r['noisy']==noisy and (r.get('model')=='candidate' if prefix=='combined_refinement' else r['alpha']==report['selected']['alpha'])]
            reg.append(dict(name=label,noisy=noisy,**aggregate(subset)))
    winner=max(eligible,key=lambda r:(r['validation']['spell_f1'],r['validation']['macro_f1']))
    assert winner['name']=='Composite'
    residual=read('external_reports/residual_missing_validation.json')
    assert not residual['accepted'],'A new qualified residual must be considered before release'
    meta=read('models/external_v5.joblib.metadata.json')
    assert meta['method']=='combined' and all(meta['gates'].values())
    summary=dict(status='Repeated external development, not fresh blind testing',winner=winner,validation=validation,
        cross_validation=cv,regression=reg,model_sha256=meta['model_sha256'],
        selection_protocol_sha256=hashlib.sha256((ROOT/'MULTIAGENT_SELECTION_PROTOCOL.md').read_bytes()).hexdigest())
    (ROOT/'external_reports/multiagent_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Paper-driven multi-agent model improvement','',
        'V5 selects the composite of observability-aware16Hz contact emissions and',
        'directed transition boundaries. It is the highest-WR9 candidate that passed',
        'all declared clean/noisy whole-match CV and reused-test regression checks.',
        'The selection rule was recorded in MULTIAGENT_SELECTION_PROTOCOL.md. V4.1',
        'is preserved as a fallback. Saved-model software checks are recorded separately.', '',
        '## What the agents contributed','',
        '- Research: audited Borghesi2023, Ferraz2025, ELASTIC2026 and PathCRF2026;',
        '  translated contact physics into tested soft features. PathCRF released',
        '  weights overlap our evaluation matches and were not used.',
        '- Audit: verified labels, timing and preprocessing; found lost raw ball',
        '  observability, rounded16Hz velocity approximations and repeated training',
        '  references. Verified new inputs and exact zero-padded initialization.',
        '- Implementation: fitted directed and physics boundary models, evaluated',
        '  symmetry averaging and a protected missing-input adapter; built and',
        '  tested the new production producer.',
        '- Integration: fitted the new neural inputs, selected and confirmed the',
        '  composite and prepared the reproducible release.', '',
        'Primary links and detailed caveats: RESEARCH_AGENT_METHODS.md and',
        'PAPER_RESEARCH_NOTES_2026_11_06.md. These are original adaptations, not',
        'reproductions of the papers private-data scores or pretrained products.', '',
        '## Reused external validation selection','',
        '| Candidate | Spell F1 | Short /188 |','|---|---:|---:|']
    for r in validation:lines.append('| {} | {:.4f} | {} |'.format(r['name'],r['spell_f1'],r['short_hits']))
    lines+=['','Native25Hz was rejected. Observability16Hz was selected before CV. The',
        'composite alpha1 was selected on WR9 before its CV scores were inspected.',
        'The residual adapter preserved all68old tensors but failed noisy validation;',
        'no CV/regression followed. Symmetry and the first directed model failed',
        'their standalone CV gates; no standalone regression was searched.', '',
        '## Whole-match comparisons','',
        '| Candidate | Tracking | Spell F1 | Short /463 |','|---|---|---:|---:|']
    for noisy in [False,True]:
        base=aggregate([r for r in composite_cv['rows'] if r['model']=='v4.1' and r['noisy']==noisy])
        lines.append('| V4.1 | {} | {:.4f} | {} |'.format('noise' if noisy else 'clean',base['spell_f1'],base['short_hits']))
    for r in cv:lines.append('| {} | {} | {:.4f} | {} |'.format(r['name'],'noise' if r['noisy'] else 'clean',r['spell_f1'],r['short_hits']))
    lines+=['','## Single-choice reused regression','',
        '| Candidate | Tracking | Spell F1 | Short /362 |','|---|---|---:|---:|']
    comp_reg=read('external_reports/combined_refinement_regression.json')
    for noisy in [False,True]:
        base=aggregate([r for r in comp_reg['rows'] if r['model']=='v4.1' and r['noisy']==noisy])
        lines.append('| V4.1 | {} | {:.4f} | {} |'.format('noise' if noisy else 'clean',base['spell_f1'],base['short_hits']))
    for r in reg:lines.append('| {} | {} | {:.4f} | {} |'.format(r['name'],'noise' if r['noisy'] else 'clean',r['spell_f1'],r['short_hits']))
    lines+=['','## Limits and integrity','',
        'All7licensed IDSSE matches have appeared in development. Both halves stay',
        'together and each learned fold component excludes the evaluated matches.',
        'Scores use the exact organizer scorer and original corruption seeds. All',
        'V4.1 controls reproduce exactly. Clean/noisy scenarios share labels. Small',
        'gains do not establish statistical significance or professional superiority.',
        'No competition inputs, video, labels or prediction quality were used.', '',
        'One damaged external parsed cache was rebuilt from checksum-verified',
        'original XML. Rebuilt clean/noisy geometry and object features match the',
        'historical caches exactly; the damaged copy was preserved. Evidence:',
        'external_reports/agent_cache_integrity_repair.json. All14external half',
        'frame-ID sequences are contiguous. See V5_MODEL_GUIDE.md for final inputs,',
        'training and inference; the physics alternative remains a research result.', '']
    text='\n'.join(lines)
    for a,b in {'observability-aware16Hz':'observability-aware 16 Hz',
        'Borghesi2023':'Borghesi (2023)', 'Ferraz2025':'Peral/Ferraz (2025)',
        'ELASTIC2026':'ELASTIC (2026)', 'PathCRF2026':'PathCRF (2026)',
        'rounded16Hz':'rounded 16 Hz', 'Native25Hz':'Native 25 Hz',
        'Observability16Hz':'Observability 16 Hz', 'alpha1':'alpha 1',
        'all68old':'all 68 old', 'All7licensed':'All 7 licensed',
        'All14external':'All 14 external', 'papers private-data':"papers' private-data"}.items():text=text.replace(a,b)
    (ROOT/'MULTIAGENT_RESULTS.md').write_text(text,encoding='utf-8')
    print('Multi-agent cycle documented; winner verified from declared validation ranking.')
