"""Local input adaptation for the contact network; geometry and metadata only."""
import xml.etree.ElementTree as ET
import numpy as np
from paper2023_data import object_features,relative_features


def goalkeeper_flags(meta,count):
    root=ET.parse(meta).getroot()
    known=set()
    names={'gk','goalkeeper','tw'}
    def label(value):return str(value).lower().replace(' ','').replace('_','').replace('-','')
    for element in root.iter():
        if element.tag.lower() in ['role','position']:
            if any(label(element.get(k,'')) in names for k in ['name','label','value']):
                if element.get('id') is not None:known.add(element.get('id'))
    flags=[]
    for player in root.iter('Player'):
        keeper=False
        for element in player.iter():
            tag=element.tag.lower()
            if tag in ['role','position','roleref','positionref']:
                if element.get('id') in known or element.get('ref') in known or label(element.text or '') in names:
                    keeper=True
            if 'goalkeeper' in tag and label(element.text or '') in ['1','true','yes']:
                keeper=True
            for key,value in element.attrib.items():
                field=key.lower()
                if 'goalkeeper' in field and label(value) in ['1','true','yes']:
                    keeper=True
                if field in ['role','rolename','position','positionname','playingposition','type'] and label(value) in names:
                    keeper=True
                if field in ['roleid','positionid'] and value in known:
                    keeper=True
        flags.append(keeper)
    if len(flags)!=count:raise ValueError('Player role metadata disagrees with tracking slots')
    return np.asarray(flags,dtype=bool)


def contact_features(players,teams,ball,pitch,keepers,source_fps=25):
    """Retain every supplied slot, padding in 17-player groups per team.

    The contact network masks missing objects and is invariant to additional
    missing padding at inference. No player is discarded to enforce 36 slots.
    """
    ids=[np.flatnonzero(teams==t) for t in [0,1]]
    groups=max(1,(max(map(len,ids))+16)//17)
    blocks=[]
    for group in range(groups):
        take=np.r_[ids[0][group*17:(group+1)*17],ids[1][group*17:(group+1)*17]]
        x,samples=object_features(players[:,take],teams[take],ball,pitch,np.asarray(keepers)[take],
                                   source_fps=source_fps,interpolation='short')
        blocks.append(x)
    x=np.concatenate([blocks[0][:,:2]]+[x[:,2:19] for x in blocks]+[x[:,19:36] for x in blocks],axis=1)
    return relative_features(x),samples


def infer_goalkeepers(players,teams,pitch):
    """Fixed geometry fallback for absent roles, with no fitting or labels.

    Keep tracks whose median position is in an end penalty area and within
    five metres of the deepest observed teammate. Track fragments are allowed.
    """
    median_x=np.full(players.shape[1],-np.inf)
    median_y=np.full(players.shape[1],np.inf)
    for i in range(players.shape[1]):
        seen=np.isfinite(players[:,i]).all(1)
        if seen.sum()>=25:
            median_x[i],median_y[i]=np.median(np.abs(players[seen,i]),axis=0)
    result=np.zeros(players.shape[1],bool)
    for team in [0,1]:
        take=teams==team
        if take.any():
            deepest=np.max(median_x[take])
            result[take]=(median_x[take]>=deepest-5)&(median_x[take]>=pitch[0]/2-16.5)&(median_y[take]<=20.16)
    return result
