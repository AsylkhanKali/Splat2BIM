"""Initialize scale from inter-fragment spacing instead of whole-scene extent."""
import numpy as np
from scipy.spatial import cKDTree

from align_0929 import similarity
from object_consensus import descriptors,match,spatial_support

PARAMETERS={'min_target_spacing_m':20.,'size_ratio_min':.5,'size_ratio_max':2.,
    'proxy_center_gate_m':5.,'seeds_per_source_pair':4,'centroid_refit_iterations':3}


def plane_frame(normals):
    up=np.median(normals,axis=0);up/=np.linalg.norm(up)
    axis=np.eye(3)[np.argmin(abs(up))];x=axis-up*(axis@up);x/=np.linalg.norm(x)
    return np.c_[x,np.cross(up,x),up]


def from_pair(source,target,source_frame,target_frame):
    """A proper Sim(3) from two centers and oriented surface normals."""
    sv=(source[1]-source[0])@source_frame;tv=(target[1]-target[0])@target_frame
    if np.linalg.norm(sv[:2])<1e-9 or np.linalg.norm(tv[:2])<1e-9:return None
    angle=np.arctan2(tv[1],tv[0])-np.arctan2(sv[1],sv[0]);c,s=np.cos(angle),np.sin(angle)
    scale=np.linalg.norm(tv[:2])/np.linalg.norm(sv[:2])
    matrix=np.eye(4);matrix[:3,:3]=scale*target_frame@np.array([[c,-s,0],[s,c,0],[0,0,1.]])@source_frame.T
    matrix[:3,3]=target.mean(0)-matrix[:3,:3]@source.mean(0)
    return matrix


def hypotheses(source_found,target_found):
    source=descriptors(source_found);target=descriptors(target_found);p=PARAMETERS
    if len(source['ids'])<3 or len(target['ids'])<3:return [],{'pair_trials':0,'retained_proposals':0}
    sf=plane_frame(source['normals']);tf=plane_frame(target['normals'])
    sc=source['centers']@sf;tc=target['centers']@tf
    i,j=np.triu_indices(len(tc),1);ti=np.r_[i,j];tj=np.r_[j,i]
    vector=tc[tj,:2]-tc[ti,:2];distance=np.linalg.norm(vector,axis=1)
    keep=distance>=p['min_target_spacing_m'];ti,tj,vector,distance=ti[keep],tj[keep],vector[keep],distance[keep]
    angle=np.arctan2(vector[:,1],vector[:,0]);mid=(tc[ti]+tc[tj])/2
    tree=cKDTree(tc);proposals=[];trials=0
    for si,sj in zip(*np.triu_indices(len(sc),1)):
        v=sc[sj,:2]-sc[si,:2];length=np.linalg.norm(v)
        if length<1e-9:continue
        scales=distance/length
        ratios=np.stack([source['lengths'][si]*scales[:,None]/target['lengths'][ti],
            source['lengths'][sj]*scales[:,None]/target['lengths'][tj]],axis=1)
        eligible=np.all((ratios>=p['size_ratio_min'])&(ratios<=p['size_ratio_max']),axis=(1,2))
        ids=np.flatnonzero(eligible);trials+=len(ids)
        if not len(ids):continue
        selected=[]
        for start in range(0,len(ids),512):
            batch=ids[start:start+512];scale=scales[batch];yaw=angle[batch]-np.arctan2(v[1],v[0])
            c,s=np.cos(yaw),np.sin(yaw);q=sc-(sc[si]+sc[sj])/2
            moved=np.empty((len(batch),len(sc),3))
            moved[:,:,0]=(c[:,None]*q[:,0]-s[:,None]*q[:,1])*scale[:,None]+mid[batch,0,None]
            moved[:,:,1]=(s[:,None]*q[:,0]+c[:,None]*q[:,1])*scale[:,None]+mid[batch,1,None]
            moved[:,:,2]=q[:,2]*scale[:,None]+mid[batch,2,None]
            error,near=tree.query(moved);valid=error<=p['proxy_center_gate_m']
            size=source['lengths'][None,:,:]*scale[:,None,None]/target['lengths'][near]
            valid&=np.all((size>=p['size_ratio_min'])&(size<=p['size_ratio_max']),axis=2)
            # Count each target at most once in this fast shortlist score.
            unique=np.sort(np.where(valid,near,len(tc)),axis=1)
            count=np.sum((unique!=len(tc))&np.c_[np.ones(len(batch),bool),np.diff(unique,axis=1)!=0],axis=1)
            mean=np.sum(np.where(valid,error,0),axis=1)/np.maximum(valid.sum(1),1)
            order=np.lexsort((mean,-count))[:p['seeds_per_source_pair']]
            selected.extend((int(count[k]),float(mean[k]),int(batch[k])) for k in order if count[k]>=3)
        selected=sorted(selected,key=lambda x:(-x[0],x[1],x[2]))[:p['seeds_per_source_pair']]
        for count,error,k in selected:
            matrix=from_pair(source['centers'][[si,sj]],target['centers'][[ti[k],tj[k]]],sf,tf)
            pairs=match(source,target,matrix)
            if not spatial_support(pairs,source,target,matrix):continue
            for _ in range(p['centroid_refit_iterations']):
                fitted=similarity(source['centers'][[r['source_index'] for r in pairs]],target['centers'][[r['target_index'] for r in pairs]],True)
                new=match(source,target,fitted)
                if not spatial_support(new,source,target,fitted):break
                matrix,pairs=fitted,new
            proposals.append({'matrix':matrix,'pairs':pairs,'anchor_source_ids':[source['ids'][si],source['ids'][sj]],
                'anchor_target_ids':[target['ids'][ti[k]],target['ids'][tj[k]]],
                'proxy_unique_targets':count,'proxy_mean_m':error})
    return proposals,{'pair_trials':trials,'retained_proposals':len(proposals)}
