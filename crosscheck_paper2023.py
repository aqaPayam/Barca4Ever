"""Compare externally retrained paper models with frozen V3 on identical folds."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from train_paper2023 import load_model
from paper2023_data import get
from paper2023_model import outputs,upsample,joint_probabilities,hierarchical_labels
from external_idsse import SPLITS
from learned_possession import viterbi
from develop_v2 import score
from torch_runtime import torch

ROOT=Path(__file__).resolve().parent


def prediction(marginals,penalty):
    return hierarchical_labels(marginals) if penalty=='hierarchical' else viterbi(joint_probabilities(marginals),penalty)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipes',nargs='+',default=['paper','adapted','relative'])
    parser.add_argument('--out',default='external_reports/paper2023_crosscheck.json')
    args=parser.parse_args()
    target=ROOT/args.out
    if target.exists():raise RuntimeError('Preserve original comparison evidence')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    folds=[(SPLITS['train'][:2],SPLITS['train'][2:]),(SPLITS['train'][2:],SPLITS['train'][:2])]
    existing=json.loads((ROOT/'external_reports/research_v3/boundary_crosscheck.json').read_text())['rows']
    rows=[dict(r,model='v3') for r in existing if r['model']=='learned_boundary']
    selection={}
    recipes=args.recipes
    for recipe in recipes:
        report=json.loads((ROOT/f'models/paper2023/{recipe}.validation.json').read_text())
        artifact=torch.load(ROOT/f'models/paper2023/{recipe}.pt',map_location='cpu',weights_only=True)
        selection[recipe]=dict(epoch=report['selected_epoch'],width=report['architecture']['width'],
                               penalty=report['selected']['penalty'],batch=artifact['run_info']['batch_size'])
    for fold,(train,test) in enumerate(folds):
        for recipe in recipes:
            choice=selection[recipe]
            path=ROOT/f'models/paper2023/{recipe}_fold{fold+1}.pt'
            if not path.exists():
                subprocess.run([sys.executable,str(ROOT/'train_paper2023.py'),'--recipe',recipe,
                    '--matches',*train,'--fixed','--epochs',str(choice['epoch']),
                    '--width',str(choice['width']),'--batch',str(choice['batch']),'--out',str(path)],check=True,cwd=ROOT)
            model,artifact=load_model(path,device)
            assert artifact['training_matches']==train
            for m,match in enumerate(test):
                for h in [0,1]:
                    for noisy in [False,True]:
                        seed=11200+100*fold+10*m+h
                        d=get(match,h,noisy,seed=seed,interpolation=artifact['interpolation'])
                        if recipe=='relative':
                            from paper2023_data import relative_features
                            d['x']=relative_features(d['x'])
                        marginal=upsample(outputs(model,d['x'],device),d['samples'],len(d['full_y']))
                        methods=[(recipe,choice['penalty'])]
                        if recipe=='paper':methods.append(('paper_raw','hierarchical'))
                        for name,penalty in methods:
                            p=prediction(marginal,penalty);y=d['full_y'];valid=y<3
                            r=dict(fold=fold+1,match=match,half=h+1,noisy=noisy,model=name,
                                   global_accuracy=float((p[valid]==y[valid]).mean()),**score(y,p))
                            rows.append(r)
                            print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
            del model
            if device=='cuda':torch.cuda.empty_cache()
    target.write_text(json.dumps(dict(status='Development cross-validation on previously used matches; not a fresh blind test',
                        provenance='Independent reimplementation, not authors original weights',selection=selection,rows=rows),indent=2))
