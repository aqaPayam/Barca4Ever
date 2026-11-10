"""Small bounded reliability gates, fitted only on external calibration rows."""
import numpy as np
import learned_possession
from scipy.optimize import minimize
from scipy.special import expit


def reliability(tree,neural,x):
    """H/A-invariant statistics; x is the existing 59-dimensional tracking input."""
    parts=[]
    for p in [tree,neural]:
        ordered=np.sort(p,axis=1)
        alive=np.maximum(p[:,:2].sum(1),1e-7)
        parts.extend([-np.sum(p*np.log(np.clip(p,1e-7,1)),axis=1),
                      ordered[:,-1],ordered[:,-1]-ordered[:,-2],p[:,2],
                      abs(p[:,0]-p[:,1])/alive])
    parts.extend([np.sum(tree*neural,axis=1),np.abs(tree-neural).sum(1),
                  tree[:,2]-neural[:,2]])
    # Symmetric nearest-contact geometry and visibility, no names or labels.
    parts.extend([np.minimum(x[:,0],x[:,3]),abs(x[:,6]),x[:,7],
                  np.minimum(x[:,8],x[:,9]),np.maximum(x[:,8],x[:,9]),
                  x[:,10],x[:,11],x[:,12],x[:,40],x[:,41],x[:,57],x[:,58],
                  np.minimum(x[:,42],x[:,45]),abs(x[:,48])])
    parts.extend([(parts[0]-parts[5])*x[:,7],(tree[:,2]-neural[:,2])*(1-x[:,7])])
    return np.clip(np.nan_to_num(np.column_stack(parts),nan=0,posinf=100,neginf=-100),-100,100).astype(np.float64)


def objective(beta,z,t,n,weight,l2=.02):
    s=expit(z@beta)
    w=.1+.8*s
    q=np.clip(t+w*(n-t),1e-9,1)
    value=-np.sum(weight*np.log(q))/weight.sum()+.5*l2*np.dot(beta,beta)
    derivative=-weight*(n-t)/q*.8*s*(1-s)/weight.sum()
    gradient=z.T@derivative+l2*beta
    return float(value),gradient


def fit_head(f,t,n,weight):
    center=f.mean(0);scale=f.std(0)
    scale=np.where(scale<1e-6,1,scale)
    z=np.column_stack([np.ones(len(f)),np.clip((f-center)/scale,-6,6)])
    weight=weight/weight.mean()
    result=minimize(objective,np.zeros(z.shape[1]),args=(z,t,n,weight),jac=True,
                    method='L-BFGS-B',options=dict(maxiter=300,ftol=1e-11,gtol=1e-7))
    if not result.success:raise RuntimeError('Gate optimization failed: '+result.message)
    return dict(center=center,scale=scale,beta=result.x,optimizer=dict(success=bool(result.success),
                iterations=int(result.nit),objective=float(result.fun)))


def head_weight(head,f):
    z=np.column_stack([np.ones(len(f)),np.clip((f-head['center'])/head['scale'],-6,6)])
    return .1+.8*expit(z@head['beta'])


def conditional_team(p):
    return (p[:,:2]+1e-9)/(p[:,:2].sum(1,keepdims=True)+2e-9)


def fit_gate(data,kind):
    f=[];trees=[];neurals=[];labels=[];weights=[]
    from possession_v2 import spell_weights
    for d in data:
        take=(np.arange(len(d['y']))%5==0)&(d['y']<3)
        f.append(d['f'][take]);trees.append(d['tree'][take]);neurals.append(d['neural'][take])
        labels.append(d['y'][take]);weights.append(spell_weights(d['y'])[take])
    f,t,n,y,weight=map(np.concatenate,[f,trees,neurals,labels,weights])
    rows=np.arange(len(y))
    if kind=='joint':
        return dict(kind=kind,heads=[fit_head(f,t[rows,y],n[rows,y],weight)])
    assert kind=='factorized'
    dead=y==2
    td=np.where(dead,t[:,2],1-t[:,2]);nd=np.where(dead,n[:,2],1-n[:,2])
    alive=~dead
    tc=conditional_team(t)
    nc=conditional_team(n)
    return dict(kind=kind,heads=[fit_head(f,td,nd,weight),
        fit_head(f[alive],tc[rows[alive],y[alive]],nc[rows[alive],y[alive]],weight[alive])])


def combine(gate,tree,neural,f):
    w=head_weight(gate['heads'][0],f)
    if gate['kind']=='joint':return (1-w[:,None])*tree+w[:,None]*neural
    assert gate['kind']=='factorized'
    dead=(1-w)*tree[:,2]+w*neural[:,2]
    team_weight=head_weight(gate['heads'][1],f)[:,None]
    team=(1-team_weight)*conditional_team(tree)+team_weight*conditional_team(neural)
    return np.column_stack([(1-dead[:,None])*team,dead])
