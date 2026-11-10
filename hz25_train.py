"""Fixed external-only fine-tuning of observability/rate-adapted contact TCN."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from external_idsse import SPLITS
from hz25_data import get
from hz25_model import expand
from train_paper2023 import sample_refs,make_batch
from spell_sampling import spell_refs
from possession_v2 import spell_weights
from paper2023_model import joint_loss
from torch_runtime import torch

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rate',type=int,choices=[16,25],required=True)
    parser.add_argument('--matches',nargs='+',default=SPLITS['train'])
    parser.add_argument('--initialize',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();target=ROOT/args.out
    if target.exists():raise RuntimeError('Preserve prior fine-tune evidence')
    assert set(args.matches)<=set(SPLITS['train'])
    torch.set_num_threads(4);torch.manual_seed(20261116);rng=np.random.default_rng(20261116)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.manual_seed_all(20261116)
    parent=torch.load(ROOT/args.initialize,map_location='cpu',weights_only=True)
    assert parent['training_matches']==args.matches and parent['run_info']['role_mode']=='none'
    model,architecture=expand(parent);model=model.to(device)
    sequences=[]
    for match in args.matches:
        for h in [0,1]:
            for noisy in [False,True]:
                d=get(match,h,noisy,args.rate)
                assert not d['x'][:,:,6].any()
                d['weights']=spell_weights(d['full_y'])[d['samples']]
                d['valid']=np.flatnonzero((np.arange(len(d['y']))>=47)&(np.arange(len(d['y']))<len(d['y'])-16)&(d['y']<3))
                sequences.append(d)
        print('External observability input loaded:',match,args.rate,flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    batches=(100000+255)//256
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2*batches,eta_min=.00003)
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda')
    history=[];started=time.monotonic()
    for epoch in [1,2]:
        refs=sample_refs(sequences,100000,rng)
        refs[-25000:]=spell_refs(sequences,25000,rng)
        rng.shuffle(refs);model.train();losses=[]
        for start in range(0,len(refs),256):
            x,y,w=make_batch(sequences,refs[start:start+256],device,True,rng)
            with torch.autocast(device_type=device,enabled=device=='cuda'):
                dead,home=model(x);loss=joint_loss(dead.float(),home.float(),y,w)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite adaptation loss')
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer);scaler.update();scheduler.step();losses.append(loss.item())
            if start and start%(256*200)==0:print('Fine-tune progress:',args.rate,epoch,start,round(time.monotonic()-started),flush=True)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-started,
                 reference_sha256=hashlib.sha256(refs.tobytes()).hexdigest(),
                 reference_content_sha256=hashlib.sha256(refs[np.lexsort((refs[:,1],refs[:,0]))].tobytes()).hexdigest())
        history.append(row);print(json.dumps(row),flush=True)
    assert history[0]['reference_sha256']!=history[1]['reference_sha256']
    assert history[0]['reference_content_sha256']!=history[1]['reference_content_sha256']
    target.parent.mkdir(parents=True,exist_ok=True)
    artifact=dict(kind='observability_contact',state_dict={k:v.cpu() for k,v in model.state_dict().items()},
        architecture=architecture,rate=args.rate,training_matches=args.matches,epochs=2,seed=20261116,
        history=history,role_mode='none',initializer_sha256=hashlib.sha256((ROOT/args.initialize).read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['hz25_train.py','hz25_data.py','hz25_model.py','paper2023_model.py','paper2023_data.py','train_paper2023.py','spell_sampling.py','possession_v2.py','learned_possession.py']})
    torch.save(artifact,target)
    target.with_suffix('.fit.json').write_text(json.dumps({k:v for k,v in artifact.items() if k!='state_dict'},indent=2)+'\n')
    print('External observability adaptation saved:',target.name,flush=True)
