"""Label-free frozen-network symmetry audit on64 external WR9 windows.

Reads only the x array from the licensed external feature caches. No labels,
quality scores, fitting, selection or competition inputs are used. CPU only.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from paper2023_data import relative_features
from paper2023_model import PaperPossession,joint_probabilities
from torch_runtime import torch

ROOT=Path(__file__).resolve().parent
SEED=20261117


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reflected(raw):
    result=raw.copy()
    seen=raw[:,:,10]==0
    velocity_seen=seen&np.r_[seen[:1],seen[:-1]]
    result[:,:,:2][seen]*=-1
    result[:,:,2:4][velocity_seen]*=-1
    return result


def swapped(x):
    # Match training augmentation: exchange team slots and H/A role flags.
    order=np.r_[0,1,np.arange(19,36),np.arange(2,19)]
    result=x[:,:,order].copy()
    result[:,:,:,7:9]=result[:,:,:,7:9][:,:,:,::-1]
    return result


def predict(model,windows):
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(windows),8):
            dead,home=model(torch.from_numpy(windows[start:start+8]))
            marginal=torch.stack([dead.sigmoid(),home.sigmoid()],dim=1).numpy()
            chunks.append(joint_probabilities(marginal))
    result=np.concatenate(chunks)
    assert np.isfinite(result).all() and np.allclose(result.sum(1),1,atol=1e-6)
    return result


def disagreement(base,transformed):
    delta=np.abs(base-transformed)
    return dict(mean_absolute_probability_difference=float(delta.mean()),
        maximum_absolute_probability_difference=float(delta.max()),
        mean_total_variation=float((.5*delta.sum(1)).mean()),
        maximum_total_variation=float((.5*delta.sum(1)).max()),
        changed_top_label_count=int((base.argmax(1)!=transformed.argmax(1)).sum()),
        window_count=len(base),
        per_class_mean_absolute_difference=dict(zip(['H','A','D'],map(float,delta.mean(0)))))


if __name__=='__main__':
    target=ROOT/'external_reports/agent_symmetry_audit.json'
    if target.exists():raise RuntimeError('Preserve existing label-free audit evidence')
    torch.set_num_threads(2)
    model_path=ROOT/'models/paper2023/rolefree.pt'
    artifact=torch.load(model_path,map_location='cpu',weights_only=True)
    assert artifact['architecture']['channels']==17
    assert 'J03WR9' not in artifact['training_matches']
    model=PaperPossession(**artifact['architecture']).cpu().eval()
    model.load_state_dict(artifact['state_dict'])
    rng=np.random.default_rng(SEED)
    original=[];rotated=[];references=[];cache_hashes={}
    for h in [0,1]:
        path=ROOT/f'external_cache/paper2023/J03WR9_{h}_0_{8400+h}_short_v1.npz'
        with np.load(path) as cache:
            raw=cache['x'].copy()  # Deliberately do not read y or full_y.
        raw[:,:,6]=0
        ref=relative_features(raw)
        rotation=relative_features(reflected(raw))
        indices=rng.choice(np.arange(47,len(raw)-16),32,replace=False)
        for index in indices:
            original.append(ref[index-47:index+17])
            rotated.append(rotation[index-47:index+17])
            references.append((h,int(index)))
        cache_hashes[path.name]=digest(path)
    original=np.stack(original).astype(np.float32)
    rotated=np.stack(rotated).astype(np.float32)
    assert original.shape==(64,64,36,17)
    base=predict(model,original)
    swap=predict(model,swapped(original))[:,[1,0,2]]
    reflection=predict(model,rotated)
    report=dict(status='Descriptive frozen-network symmetry diagnostic only; no labels loaded or score fitting',
        device='cpu',torch_threads=2,seed=SEED,windows=64,windows_per_external_half=32,
        window_reference_sha256=hashlib.sha256(np.asarray(references,dtype=np.int64).tobytes()).hexdigest(),
        model_sha256=digest(model_path),external_cache_sha256=cache_hashes,
        source_sha256={name:digest(ROOT/name) for name in ['audit_symmetry.py','paper2023_data.py','paper2023_model.py']},
        transformations=dict(team_swap='Exchange17-slot team groups and H/A role flags; restore H/A output columns',
            pitch_reflection='Rotate pitch180degrees: negate observed XY/velocity and recompute relative features; preserve missing sentinels and team roles'),
        team_swap=disagreement(base,swap),pitch_reflection=disagreement(base,reflection),
        caveat='WR9 was used in prior development selection; disagreement indicates unlearned symmetry, not an achievable accuracy gain. Ensemble quality requires separate scoring.')
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['device','windows','team_swap','pitch_reflection']}),flush=True)
