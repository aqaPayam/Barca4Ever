"""Select a predeclared observability recipe using external WR9 only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    target=ROOT/'external_reports/hz25_selection.json'
    if target.exists():raise RuntimeError('Preserve selection evidence')
    base=json.loads((ROOT/'external_reports/v4_decoder_validation.json').read_text())['selected']
    trials=[]
    for rate in [16,25]:trials.extend(json.loads((ROOT/f'external_reports/hz25_validation_{rate}.json').read_text())['trials'])
    eligible=[r for r in trials if r['spell_f1']>base['spell_f1'] and r['short_hits']>=base['short_hits']]
    selected=max(eligible,key=lambda r:(r['spell_f1'],r['macro_f1'])) if eligible else None
    if selected:selected={k:v for k,v in selected.items() if k!='rows'}
    target.write_text(json.dumps(dict(status='External reused WR9 selection only',baseline={k:v for k,v in base.items() if k!='rows'},selected=selected,trials=trials),indent=2)+'\n')
    print('Observability selection:',json.dumps(selected),flush=True)
