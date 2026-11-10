"""External-only missing-input residual adaptation with exact legacy preservation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from external_idsse import SPLITS
from residual_missing_data import get
from residual_missing_model import initialize,restore_protected_columns,protected_proof,extra_norm,INPUT_WEIGHTS
from train_paper2023 import sample_refs,make_batch
from spell_sampling import spell_refs
from possession_v2 import spell_weights
from paper2023_model import joint_loss
from torch_runtime import torch

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--matches',nargs='+',default=SPLITS['train'])
    parser.add_argument('--initialize',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--batch',type=int,choices=[128,256],default=256)
    args=parser.parse_args();target=ROOT/args.out
    if target.exists():raise RuntimeError('Preserve residual adaptation evidence')
    assert set(args.matches)<=set(SPLITS['train'])
    torch.set_num_threads(4);torch.manual_seed(20261117);rng=np.random.default_rng(20261117)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.manual_seed_all(20261117)
    parent=torch.load(ROOT/args.initialize,map_location='cpu',weights_only=True)
    assert parent['training_matches']==args.matches and parent['run_info']['role_mode']=='none'
    model,architecture,protected,handles=initialize(parent,device)
    initial_proof=protected_proof(model,parent)
    sequences=[]
    for match in args.matches:
        for h in [0,1]:
            for noisy in [False,True]:
                d=get(match,h,noisy)
                assert not d['x'][:,:,6].any()
                d['weights']=spell_weights(d['full_y'])[d['samples']]
                d['valid']=np.flatnonzero((np.arange(len(d['y']))>=47)&(np.arange(len(d['y']))<len(d['y'])-16)&(d['y']<3))
                sequences.append(d)
        print('External residual inputs loaded:',match,flush=True)
    parameters=dict(model.named_parameters())
    optimizer=torch.optim.Adam([parameters[name] for name in INPUT_WEIGHTS],lr=.0005,weight_decay=0)
    batches=(100000+args.batch-1)//args.batch
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2*batches,eta_min=.0001)
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda')
    history=[];started=time.monotonic()
    print('Residual training:',device,'batch',args.batch,'effective_trainable_columns',sum(parameters[n][:,17:].numel() for n in INPUT_WEIGHTS),flush=True)
    for epoch in [1,2]:
        refs=sample_refs(sequences,100000,rng);refs[-25000:]=spell_refs(sequences,25000,rng)
        content_refs=refs[np.lexsort((refs[:,1],refs[:,0]))]
        rng.shuffle(refs);losses=[]
        assert not model.training
        for start in range(0,len(refs),args.batch):
            x,y,w=make_batch(sequences,refs[start:start+args.batch],device,True,rng)
            with torch.autocast(device_type=device,enabled=device=='cuda'):
                dead,home=model(x);loss=joint_loss(dead.float(),home.float(),y,w)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite residual loss')
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameters[n] for n in INPUT_WEIGHTS],5)
            scaler.step(optimizer);scaler.update();restore_protected_columns(model,protected);scheduler.step()
            losses.append(loss.item())
            if start and start%(args.batch*200)==0:print('Residual fine-tune progress:',epoch,start,round(time.monotonic()-started),flush=True)
        proof=protected_proof(model,parent)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-started,
            reference_content_sha256=hashlib.sha256(content_refs.tobytes()).hexdigest(),protected_proof=proof,
            new_columns_l2=extra_norm(model))
        history.append(row);print(json.dumps(row),flush=True)
    assert history[0]['reference_content_sha256']!=history[1]['reference_content_sha256']
    assert history[-1]['new_columns_l2']>0
    final_proof=protected_proof(model,parent)
    target.parent.mkdir(parents=True,exist_ok=True)
    artifact=dict(kind='residual_missing_contact',input_mode='missing20',rate=16,
        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},architecture=architecture,
        training_matches=args.matches,epochs=2,seed=20261117,batch_size=args.batch,role_mode='none',
        initializer_sha256=hashlib.sha256((ROOT/args.initialize).read_bytes()).hexdigest(),
        initial_proof=initial_proof,final_proof=final_proof,history=history,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            'residual_missing_train.py','residual_missing_data.py','residual_missing_model.py','RESIDUAL_MISSING_PROTOCOL.md',
            'hz25_model.py','paper2023_model.py','paper2023_data.py','train_paper2023.py','spell_sampling.py',
            'possession_v2.py','learned_possession.py','external_idsse.py']})
    torch.save(artifact,target)
    target.with_suffix('.fit.json').write_text(json.dumps({k:v for k,v in artifact.items() if k!='state_dict'},indent=2)+'\n')
    print('Residual adaptation saved; every old tensor and buffer is bit-exact.',flush=True)
