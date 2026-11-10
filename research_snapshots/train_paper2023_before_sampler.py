"""Train the paper-based model using licensed external matches only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from paper2023_data import get,relative_features, WINDOW, TARGET
from paper2023_model import PaperPossession,joint_loss,outputs,upsample,joint_probabilities,hierarchical_labels
from torch_runtime import torch
from possession_v2 import spell_weights
from learned_possession import viterbi
from external_idsse import SPLITS
from develop_v2 import score

ROOT=Path(__file__).resolve().parent


def load_model(path,device):
    a=torch.load(path,map_location='cpu',weights_only=True)
    model=PaperPossession(**a['architecture']).to(device)
    model.load_state_dict(a['state_dict']);model.eval()
    return model,a


def prepare(matches,recipe,interpolation='akima'):
    sequences=[]
    for match in matches:
        for h in [0,1]:
            for noisy in ([False] if recipe=='paper' else [False,True]):
                d=get(match,h,noisy,interpolation=interpolation)
                if recipe=='relative':d['x']=relative_features(d['x'])
                # Training weights use the original 25-Hz label durations.
                d['weights']=spell_weights(d['full_y'])[d['samples']]
                d['valid']=np.flatnonzero((np.arange(len(d['y']))>=47)&
                    (np.arange(len(d['y']))<len(d['y'])-16)&(d['y']<3))
                sequences.append(d)
        print('Loaded external trajectory training',match,flush=True)
    return sequences


def make_batch(sequences,refs,device,swap=False,rng=None):
    xx,yy,ww=[],[],[]
    for sequence,target in refs:
        d=sequences[int(sequence)];target=int(target)
        x=d['x'][target-47:target+17].copy();y=int(d['y'][target])
        if swap and rng.random()<.5:
            x=x[:,np.r_[0,1,np.arange(19,36),np.arange(2,19)]]
            x[:,:,7:9]=x[:,:,7:9][:,:,::-1]
            if y<2:y=1-y
        xx.append(x);yy.append(y);ww.append(d['weights'][target])
    return (torch.from_numpy(np.stack(xx)).to(device),
            torch.tensor(yy,dtype=torch.long,device=device),
            torch.tensor(ww,dtype=torch.float32,device=device))


def sample_refs(sequences,count,rng):
    # Uniform over eligible frames, not uniform over arbitrary chunk lengths.
    sizes=np.array([len(d['valid']) for d in sequences]);ends=np.cumsum(sizes)
    flat=rng.choice(int(ends[-1]),size=count,replace=count>ends[-1])
    seq=np.searchsorted(ends,flat,side='right');starts=np.r_[0,ends[:-1]]
    target=np.array([sequences[s]['valid'][i-starts[s]] for s,i in zip(seq,flat)])
    return np.column_stack([seq,target])


def evaluate(model,device,interpolation='akima',relative=False):
    result=[]
    for h in [0,1]:
        for noisy in [False,True]:
            d=get('J03WR9',h,noisy,interpolation=interpolation)
            if relative:d['x']=relative_features(d['x'])
            marginal=upsample(outputs(model,d['x'],device),d['samples'],len(d['full_y']))
            result.append((h,noisy,d,marginal))
    trials=[]
    for penalty in ['hierarchical',0,2,4,8,16]:
        rows=[]
        for h,noisy,d,marginal in result:
            p=(hierarchical_labels(marginal) if penalty=='hierarchical' else
               viterbi(joint_probabilities(marginal),penalty))
            y=d['full_y'];valid=y<3
            rows.append(dict(half=h+1,noisy=noisy,global_accuracy=float((p[valid]==y[valid]).mean()),
                             **score(y,p)))
        trials.append(dict(penalty=penalty,spell_f1=float(np.mean([r['spell_f1'] for r in rows])),
                           macro_f1=float(np.mean([r['macro_f1'] for r in rows])),rows=rows))
    return trials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',choices=['paper','adapted','relative'],default='paper')
    parser.add_argument('--matches',nargs='+',default=SPLITS['train'])
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--samples',type=int,default=100000)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20261107)
    parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--out')
    args=parser.parse_args()
    out=Path(args.out or f'models/paper2023/{args.recipe}.pt')
    if out.exists():raise RuntimeError('Preserve the prior trained artifact; use a new output path')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.manual_seed_all(args.seed)
    interpolation='short' if args.recipe=='relative' else 'akima'
    sequences=prepare(args.matches,args.recipe,interpolation)
    references=sample_refs(sequences,args.samples,rng)
    validation=[]
    if not args.fixed:
        for h in [0,1]:
            d=get('J03WR9',h,False,interpolation=interpolation)
            if args.recipe=='relative':d['x']=relative_features(d['x'])
            d['weights']=np.ones(len(d['y']),np.float32)
            d['valid']=np.flatnonzero((np.arange(len(d['y']))>=47)&
                           (np.arange(len(d['y']))<len(d['y'])-16)&(d['y']<3))
            validation.append(d)
        valrefs=sample_refs(validation,5000,np.random.default_rng(19021))
    architecture=dict(width=args.width,dropout=.1,channels=17 if args.recipe=='relative' else 11)
    model=PaperPossession(**architecture).to(device)
    run_info=dict(device=device,torch_version=str(torch.__version__),
                  hardware=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
                  samples_per_epoch=args.samples,batch_size=args.batch,
                  source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                     for f in ['paper2023_model.py','paper2023_data.py','train_paper2023.py']})
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5 if args.recipe=='paper' else .001)
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda')
    batch_count=(len(references)+args.batch-1)//args.batch
    scheduler=(torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.7) if args.recipe=='paper'
               else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs*batch_count,eta_min=.00003))
    history=[];best=(-1.,-1.);started=time.monotonic();all_trials=[]
    print('Training device',device,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();rng.shuffle(references);losses=[]
        for start in range(0,len(references),args.batch):
            x,y,w=make_batch(sequences,references[start:start+args.batch],device,
                             args.recipe!='paper',rng)
            with torch.autocast(device_type=device,enabled=device=='cuda'):
                dead,home=model(x)
                loss=joint_loss(dead.float(),home.float(),y,w if args.recipe!='paper' else None)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward()
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer);scaler.update()
            if args.recipe!='paper':scheduler.step()
            losses.append(loss.item())
            if start and start%(args.batch*200)==0:
                print('Progress epoch',epoch,'samples',start,'loss',round(float(np.mean(losses[-100:])),4),
                      'elapsed_s',round(time.monotonic()-started),flush=True)
        if args.recipe=='paper':scheduler.step()
        model.eval();correct=0;total=0
        if not args.fixed:
            with torch.inference_mode():
                for start in range(0,len(valrefs),args.batch):
                    x,y,w=make_batch(validation,valrefs[start:start+args.batch],device)
                    a,b=model(x);pred=torch.where(a>=0,2,torch.where(b>=0,0,1))
                    correct+=int((pred==y).sum());total+=len(y)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_accuracy=correct/total if total else None,
                 elapsed_seconds=round(time.monotonic()-started,2))
        history.append(row);print(json.dumps(row),flush=True)
        metric=(correct/total,0.) if total else (epoch,0.)
        eligible=True
        if args.recipe!='paper' and not args.fixed:
            eligible=epoch in [2,4,args.epochs]
            if eligible:
                trials=evaluate(model,device,interpolation,args.recipe=='relative')
                for trial in trials:trial['epoch']=epoch
                all_trials.extend(trials)
                trial_best=max(trials,key=lambda t:(t['spell_f1'],t['macro_f1']))
                metric=(trial_best['spell_f1'],trial_best['macro_f1'])
                print('Adapted validation epoch',epoch,'best spell F1',round(metric[0],5),flush=True)
        if eligible and metric>best:
            best=metric
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                architecture=architecture,recipe=args.recipe,epoch=epoch,
                seed=args.seed,training_matches=args.matches,kind='paper2023_two_branch',
                interpolation=interpolation,history=history.copy(),run_info=run_info),out)
    artifact_hash=hashlib.sha256(out.read_bytes()).hexdigest()
    if not args.fixed:
        model,artifact=load_model(out,device)
        trials=evaluate(model,device) if args.recipe=='paper' else all_trials
        report=dict(status='External development validation; independent paper-based reimplementation, not original weights',
            recipe=args.recipe,architecture=artifact['architecture'],selected_epoch=artifact['epoch'],
            model_sha256=artifact_hash,history=history,trials=trials,
            selected=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1'])))
        report_path=out.with_suffix('.validation.json')
        report_path.write_text(json.dumps(report,indent=2))
        print('Selected external validation',json.dumps({k:v for k,v in report['selected'].items() if k!='rows'}),flush=True)
    else:
        out.with_suffix('.fit.json').write_text(json.dumps(dict(recipe=args.recipe,history=history,
            matches=args.matches,model_sha256=artifact_hash),indent=2))


if __name__=='__main__':main()
