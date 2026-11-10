"""Write the final submission report from frozen V5 evidence."""
import json
from pathlib import Path
import shutil
from report_v2 import aggregate

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    meta=json.loads((ROOT/'models/external_v5.joblib.metadata.json').read_text())
    assert meta['method']=='combined' and meta['input_mode']=='confidence20' and meta['rate']==16
    target=ROOT/'submission/Barca4Ever.md';previous=ROOT/'submission/Barca4Ever_v4_1.md'
    if not previous.exists():
        assert '(v4.1)' in target.read_text().splitlines()[0]
        shutil.copyfile(target,previous)
    lines=['# Barca4Ever - Initial Challenge technical report (V5)','',
        '## Approach and inputs','',
        'We estimate home possession H, away possession A and dead ball D from',
        'tactical player/ball XY tracking and team assignments. Long complete player',
        'tracking outages of at least10seconds receive X and split inference blocks.',
        'All fitting, hyperparameter selection and quality evaluation use licensed',
        'external data only. The supplied match is used solely for local inference.', '',
        'The unchanged strict ATD loader maps slots to teams, converts normalized',
        'XY to pitch-centred metres, and maps ATD record1 to output frame0. It',
        'rejects missing, duplicate/reordered/malformed rows and inconsistent slot',
        'counts. No player name, match ID, event outcome, clock or possession label',
        'enters a feature. Video, Smart Tagging, broadcast tracking and homography',
        'are not used. XY alone cannot identify every whistle or aerial contact.', '',
        '## Possession probabilities','',
        'The geometric branch is the preserved V2 histogram gradient boosting',
        'classifier:59 geometry/motion/visibility inputs,200iterations,23leaves,',
        'learning rate0.07,L2=3,minleaf60. Its probabilities are averaged equally',
        'with a contact-network branch extending Borghesi et al. (2023),',
        'https://doi.org/10.1016/j.eswa.2023.119780. Original author weights and',
        'confidential data were unavailable; this is an independently trained',
        'adaptation with declared assumptions, not their reported model score.', '',
        'The shared per-object trajectory TCN has width64, a1x1 stem and six causal',
        'kernel2 blocks with dilations1/2/4/8/16/32, dropout0.1 and batch normalization.',
        'Separate ball-query attention and ELU64/32 FFNs predict dead probability',
        'and conditional home probability. Home loss is zero on dead labels.',
        'Relative ball geometry/motion, a target-frame MLP and learned distance-biased',
        'attention add direct contact evidence. Missing objects are masked; every',
        'observed tracking actor is retained in padded groups, without a36-object cap.', '',
        'V5 has20input channels. The original17 are recomputed with source-rate',
        'short-gap interpolation and velocities divided by actual elapsed source',
        'time, correcting rounded16Hz interval variation. Three extra channels',
        'retain raw ball visibility and time since/until raw observation, capped',
        'at10seconds and scaled by10. Filled geometry remains separate from raw',
        'visibility. Keeper roles are always zero; player role metadata is unnecessary.', '',
        'Neural inputs use16Hz,64samples and target47 (48th element), about4seconds',
        'of trajectory with1second of future trajectory. Gap-age features may use',
        'up to10seconds of future observation evidence, so this is offline inference.',
        'The native25Hz candidate was evaluated and rejected on external validation.',
        'Marginals are interpolated to every25Hz frame; joint probabilities are',
        '[(1-dead)*home,(1-dead)*(1-home),dead].', '',
        '## Directed transition decoding','',
        'A new classifier predicts background plus six ordered transitions:',
        'H->A,H->D,A->H,A->D,D->H,D->A. It uses293 label-free inputs:59base',
        'features plus before/after/signed-difference context for26selected features',
        'at+/-2,6,12frames. Histogram boosting uses100iterations,11leaves,',
        'learning rate0.08,L2=5,minleaf80,seed20261116. Targets are nearest external',
        'changes expanded by two frames, earlier change breaking ties. Fitting',
        'samples every fifth valid frame; positive targets have weight20.', '',
        'Viterbi maximizes summed log possession probability minus switch cost',
        '16/(1+8*b[t,previous,next]). Unlike V4.1, each directed score lowers only',
        'its corresponding switch cost. Scores are uncalibrated because positives',
        'are weighted. WR9 selected pure directed evidence (alpha1) for the final',
        'composite. The old destination detector is retained in the artifact for',
        'reproduction but contributes zero at this alpha. This is an original',
        'paper-informed boundary design, not a reproduction of ASRF or ELASTIC.', '',
        'Internal ball gaps up to2seconds may be interpolated without extrapolation;',
        'longer gaps stay missing. Missing ball does not automatically mean dead.',
        'Loose balls, deflections and restarts are learned through external H/A/D',
        'labels and directed changes; there is no separate loose-ball class or',
        'manually selected restart interval from the supplied game.', '',
        '## External fitting and selection','',
        'IDSSE only: DFL/Sportec data, Bassek et al., Scientific Data12,195(2025),',
        'https://doi.org/10.1038/s41597-025-04505-y. Deposit:',
        'https://doi.org/10.6084/m9.figshare.28196177, CC BY4.0,',
        'https://creativecommons.org/licenses/by/4.0/. Preserve this attribution',
        'with the trained artifact. BallStatus0 maps to D; live BallPossession1/2',
        'maps to H/A; unknown maps to X. Labels are isolated from feature inputs.', '',
        'Base fit: J03WOH/J03WOY/J03WPY/J03WQQ; validation: J03WR9; original',
        'held-out games J03WMX/J03WN1 are now explicitly reused regression. All seven',
        'have appeared in development. Whole-match CV fits WOH/WOY and evaluates',
        'WPY/WQQ, then reverses; both halves stay together. Every neural initializer,',
        'fine-tune, tree and directed detector excludes its evaluated fold matches.', '',
        'V5 initializes from the corresponding frozen role-free checkpoint. Its',
        'new input columns are zero-padded initially. It fine-tunes two epochs of',
        '100k windows, batch256, Adam1e-4/cosine to3e-5, seed20261116. Each epoch',
        'draws new references:75%eligible frames and25%uniform team spells. Loss',
        'weights are clip(sqrt(250/duration_frames),0.5,4). Clean/noisy observations',
        'and team swaps are included. Label-independent corruption reuses the',
        'original jitter,5%player drops and15-84-frame ball gaps. Synthetic noise',
        'does not reproduce all automatic-tracking errors or identity switches.', '',
        'Rates16/25 and neural mixture weights0.25/0.5/0.75 were selected on WR9.',
        'The composite tested directed weights0/0.5/1 on WR9 before CV. Final rate16,',
        'neural weight0.5 and directed weight1 then passed the declared clean/noisy',
        'CV and regression checks. Among fully qualifying methods, the composite',
        'had the highest WR9 spell F1. Failed individual methods, alternate physics',
        'and symmetry results are retained in MULTIAGENT_RESULTS.md. No released',
        'PathCRF weights overlapping evaluated matches were used.', '',
        '## Matched external results','',
        'Exact organizer spell F1 is primary, with mean halves weighted equally.',
        'Short recovery counts same-team possessions under2seconds at IoU0.5.',
        'Clean/noisy scenarios share labels and are not independent samples.', '',
        '| Comparison | Tracking | V4.1 spell F1 | V5 spell F1 | V4.1 short | V5 short |',
        '|---|---|---:|---:|---:|---:|']
    for stage,label in [('crosscheck','Whole-match CV'),('regression','Reused regression')]:
        report=json.loads((ROOT/f'external_reports/combined_refinement_{stage}.json').read_text())
        assert report['accepted']
        for n in [False,True]:
            a,b=report['summary']['v4.1'][str(n)],report['summary']['candidate'][str(n)]
            lines.append('| {} | {} | {:.4f} | {:.4f} | {}/{} | {}/{} |'.format(label,'noise' if n else 'clean',a['spell_f1'],b['spell_f1'],a['short_hits'],a['short_count'],b['short_hits'],b['short_count']))
    lines+=['','These are small, repeated development gains, not statistical significance,',
        'a fresh blind test, a professional-system comparison or a competition score.',
        'Seven matches from one provider give limited independent evidence. Actual',
        'tracking identity switches, body orientation and aerial contacts remain',
        'limitations. Provider possession definitions may differ from Metrica.', '',
        '## Reproduce and verify','',
        'Python3.12, requirements-contact.txt. Canonical CPU inference uses4threads.',
        'The trusted artifact contains NumPy neural state plus trained tree models,',
        'with exact serialization checks and pinned model/source/evidence hashes.',
        'No external training dataset is needed to regenerate the submission.', '',
        '```text','python -m pip install -r requirements-contact.txt',
        'python refined_infer.py --model models/external_v5.joblib --data-dir /local/bundle --out Barca4Ever.csv',
        'python evaluation/validate_submission.py Barca4Ever.csv','```','',
        'The bundle contains tactical/BAR-ATH_tactical_atd.xml and its TXT. The',
        'Windows wrapper verifies the artifact checksum and stores validator details',
        'locally. The file has199,319 indexed rows. Its manifest records model/output',
        'hashes.82synthetic/unit tests, four saved-model external CPU comparisons',
        'and full-length synthetic CLI/format checks precede local inference.',
        'Format validation does not establish football quality. No fitting or',
        'adjustment happens during inference, and predictions are not checked against',
        'the supplied video or ground truth. No supplied data or extracts are sent',
        'to hosted AI services. Private inputs, intermediates, outputs and validator',
        'logs remain outside the public code package. V4.1 and older fallbacks are',
        'preserved. Details: V5_MODEL_GUIDE.md and MULTIAGENT_RESULTS.md.', '']
    text='\n'.join(lines)
    replacements={'at10seconds':'at 10 seconds','least10seconds':'least 10 seconds','record1':'record 1','frame0':'frame 0','classifier:59':'classifier: 59','200iterations':'200 iterations','23leaves':'23 leaves','rate0.07':'rate 0.07','minleaf60':'minleaf 60','width64':'width 64','a1x1':'a 1x1','kernel2':'kernel 2','dilations1':'dilations 1','dropout0.1':'dropout 0.1','ELU64/32':'ELU 64/32','20input':'20 input','original17':'original 17','rounded16Hz':'rounded 16 Hz','at10seconds':'at 10 seconds','by10':'by 10','use16Hz,64samples':'use 16 Hz, 64 samples','target47':'target 47','about4seconds':'about 4 seconds','with1second':'with 1 second','to10seconds':'to 10 seconds','native25Hz':'native 25 Hz','every25Hz':'every 25 Hz','uses293':'uses 293','59base':'59 base','26selected':'26 selected','100iterations,11leaves':'100 iterations, 11 leaves','rate0.08':'rate 0.08','minleaf80':'minleaf 80','seed20261116':'seed 20261116','weight20':'weight 20','alpha1':'alpha 1','to2seconds':'to 2 seconds','Data12,195(2025)':'Data 12, 195 (2025)','BY4.0':'BY 4.0','BallStatus0':'BallStatus 0','BallPossession1/2':'BallPossession 1/2','batch256':'batch 256','Adam1e-4':'Adam 1e-4','to3e-5':'to 3e-5','references:75%':'references: 75%','and25%':'and 25%','jitter,5%':'jitter, 5%','and15-84':'and 15-84','Rates16/25':'Rates 16/25','weights0.25':'weights 0.25','weights0/0.5/1':'weights 0/0.5/1','rate16':'rate 16','weight0.5':'weight 0.5','weight1':'weight 1','under2seconds':'under 2 seconds','IoU0.5':'IoU 0.5','Python3.12':'Python 3.12','uses4threads':'uses 4 threads','has199,319':'has 199,319','hashes.82synthetic':'hashes. 82 synthetic'}
    replacements.update({'inputs,200':'inputs, 200', 'iterations,23':'iterations, 23',
        '0.07,L2=3,minleaf':'0.07, L2=3, minleaf', 'a36-object':'a 36-object',
        'has20':'has 20', 'inputs:59':'inputs: 59', 'for26':'for 26',
        'at+/-2,6,12frames':'at +/-2, 6, 12 frames', 'uses100':'uses 100',
        '0.08,L2=5,minleaf':'0.08, L2=5, minleaf', '80,seed':'80, seed',
        '75%eligible':'75% eligible', '25%uniform':'25% uniform', '5%player':'5% player'})
    for a,b in replacements.items():text=text.replace(a,b)
    target.write_text(text,encoding='utf-8')
    print('V5 technical report generated from frozen composite evidence; V4.1 report preserved.')
