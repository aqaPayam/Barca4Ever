"""Original compact TCN with possession and boundary heads, external-only.

Inspired by Borghesi et al. 2023 (possession TCN), MS-TCN (Farha/Gall 2019),
and boundary-aware segmentation (Ishikawa et al. 2021). Not a reproduction of
those papers: uses our 59 invariant geometry features, not player/video encoders.
"""
from pathlib import Path
import sys
TORCH_DEPS=Path(__file__).resolve().parent/'.torch_deps'
if TORCH_DEPS.exists():sys.path.insert(0,str(TORCH_DEPS))
import json
import time
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from research_data import get,validation
from external_idsse import SPLITS
from learned_possession import viterbi
from research_trees import adaptive_decode
from develop_v2 import score

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'external_reports/research_v3'
MODELS=ROOT/'models/research_v3'


class Residual(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.conv=nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.mix=nn.Conv1d(width,width,1)
        self.drop=nn.Dropout(.1)
    def forward(self,x):return x+self.drop(self.mix(F.relu(self.conv(x))))


class PossessionTCN(nn.Module):
    def __init__(self,inputs=59,width=32):
        super().__init__()
        self.stem=nn.Conv1d(inputs,width,1)
        self.blocks=nn.Sequential(*[Residual(width,2**i) for i in range(5)])
        self.possession=nn.Conv1d(width,3,1)
        self.boundary=nn.Conv1d(width,1,1)
    def forward(self,x):
        z=self.blocks(F.relu(self.stem(x)))
        return self.possession(z),self.boundary(z).squeeze(1)


def boundary_target(y):
    raw=np.r_[False,y[1:]!=y[:-1]]
    target=raw.copy()
    for shift in [1,2]:
        target[shift:] |= raw[:-shift];target[:-shift] |= raw[shift:]
    return target.astype(np.float32)


def probabilities(model,x,mean,std):
    """Chunked inference with context overlap; padding matches training convention."""
    model.eval()
    x=np.clip((x-mean)/std,-8,8).astype(np.float32)
    prob=np.zeros((len(x),3),dtype=np.float32);bound=np.zeros(len(x),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(x),4096):
            end=min(len(x),start+4096);left=max(0,start-64);right=min(len(x),end+64)
            a,b=model(torch.from_numpy(x[left:right].T.copy())[None])
            sl=slice(start-left,end-left)
            prob[start:end]=a.softmax(1)[0,:,sl].T.numpy()
            bound[start:end]=b.sigmoid()[0,sl].numpy()
    return prob,bound


def load_artifact(path):
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    model=PossessionTCN();model.load_state_dict(artifact['state_dict']);model.eval()
    return model,artifact['mean'].numpy(),artifact['std'].numpy(),artifact


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--matches',nargs='+',default=SPLITS['train'])
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--fixed',action='store_true',help='Fit without validation or checkpoint selection')
    parser.add_argument('--out',default=str(MODELS/'tcn.pt'))
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261106)
    rng=np.random.default_rng(20261106)
    MODELS.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sequences=[];sample=[]
    for match in args.matches:
        print('TCN external sequences '+match,flush=True)
        for h in [0,1]:
            for noisy in [False,True]:
                for swap in [False,True]:
                    d=get(match,h,noisy,swap)
                    d['boundary']=boundary_target(d['y'])
                    sequences.append(d);sample.append(d['x'][::25])
    sample=np.concatenate(sample)
    mean=sample.mean(0);std=np.maximum(sample.std(0),.1)
    for d in sequences:d['x']=np.clip((d['x']-mean)/std,-8,8).astype(np.float32)
    val=[] if args.fixed else validation()
    model=PossessionTCN()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,1200,eta_min=.00003)
    all_trials=[];best_value=-1;started=time.monotonic()
    for step in range(1,args.steps+1):
        model.train()
        xb,yb,wb,bb=[],[],[],[]
        for _ in range(16):
            d=sequences[int(rng.integers(len(sequences)))];start=int(rng.integers(0,len(d['y'])-384))
            xb.append(d['x'][start:start+384].T)
            yb.append(d['y'][start+64:start+320])
            wb.append(d['w'][start+64:start+320])
            bb.append(d['boundary'][start+64:start+320])
        x=torch.from_numpy(np.stack(xb));y=torch.from_numpy(np.stack(yb).astype(np.int64))
        w=torch.from_numpy(np.stack(wb));target=torch.from_numpy(np.stack(bb))
        logits,bound=model(x);logits=logits[:,:,64:320];bound=bound[:,64:320]
        ce=F.cross_entropy(logits,y,reduction='none',ignore_index=3)
        loss=(ce*w).mean()+.2*F.binary_cross_entropy_with_logits(bound,target,pos_weight=torch.tensor(10.))
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step();scheduler.step()
        if step%100==0:print('TCN step {} loss {:.4f} elapsed {:.1f}s'.format(step,loss.item(),time.monotonic()-started),flush=True)
        if args.fixed or step not in [400,800,1200]:continue
        outputs=[probabilities(model,d['x'],mean,std) for h,n,d in val]
        for penalty in [2,8,16]:
            rows=[dict(half=h+1,noisy=n,**score(d['y'],viterbi(p,penalty)))
                  for (h,n,d),(p,b) in zip(val,outputs)]
            r=dict(step=step,penalty=penalty,kind='tcn',boundary_scale=0,
                   spell_f1=float(np.mean([a['spell_f1'] for a in rows])),
                   macro_f1=float(np.mean([a['macro_f1'] for a in rows])),rows=rows)
            all_trials.append(r)
            print('TCN validation '+json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
            if r['spell_f1']>best_value:
                best_value=r['spell_f1']
                torch.save(dict(state_dict=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),
                    step=step,penalty=penalty,feature_version=2,kind='tcn'),args.out)
    if args.fixed:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),
                    step=args.steps,feature_version=2,kind='tcn'),args.out)
        return
    best=max(all_trials,key=lambda r:(r['spell_f1'],r['macro_f1']))
    (REPORT/'tcn.json').write_text(json.dumps(dict(status='External validation selection, not independent testing',
                            trials=all_trials,selected=best),indent=2))


if __name__=='__main__':main()
