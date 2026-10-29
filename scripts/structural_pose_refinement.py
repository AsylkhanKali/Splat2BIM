"""Family-balanced rigid refinement on fixed, independently extracted surfaces."""
import itertools

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from align_0929 import apply

PARAMETERS={'voxel_m':.10,'maximum_points_per_surface':4000,'plane_band_normalized':.05,
    'calibration_block_m':2.,'calibration_modulus':5,'minimum_fit_points':30,'minimum_calibration_points':10,
    'normal_neighbors':24,'normal_radius_m':1.25,'minimum_normal_neighbors':8,'maximum_normal_scatter':.08,
    'huber_m':.15,'correspondence_gate_m':1.,'rotation_lever_m':100.,'svd_relative_cutoff':.01,
    'maximum_iterations':60,'maximum_motion_m':.5,'tangent_weights':[0.,.1,1.],
    'minimum_calibration_loss_improvement':1e-6,'maximum_family_median_regression_m':.01,
    'maximum_training_median_regression_m':.01}


def footprint_mask(points,found,row):
    local=((points-found['normalization_center'])*found['normalization_scale']-row['center'])@np.array(row['frame'])
    return np.all(abs(local[:,:2])<=np.array(row['lengths'])/2+1e-9,axis=1)&(abs(local[:,2])<=PARAMETERS['plane_band_normalized'])


def sample_indices(points):
    _,ids=np.unique(np.floor(points/PARAMETERS['voxel_m']).astype(int),axis=0,return_index=True)
    if len(ids)>PARAMETERS['maximum_points_per_surface']:
        ids=ids[np.linspace(0,len(ids)-1,PARAMETERS['maximum_points_per_surface'],dtype=int)]
    return np.sort(ids)


def target_surface(points):
    tree=cKDTree(points);distance,indices=tree.query(points,k=PARAMETERS['normal_neighbors'],distance_upper_bound=PARAMETERS['normal_radius_m'])
    good=np.isfinite(distance);neighbors=points[np.minimum(indices,len(points)-1)]
    count=good.sum(1);mean=np.sum(neighbors*good[:,:,None],axis=1)/np.maximum(count[:,None],1)
    centered=(neighbors-mean[:,None,:])*good[:,:,None]
    values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',centered,centered)/np.maximum(count[:,None,None],1))
    valid=(count>=PARAMETERS['minimum_normal_neighbors'])&(values[:,0]/np.maximum(values.sum(1),1e-12)<PARAMETERS['maximum_normal_scatter'])
    return {'points':points,'tree':tree,'normals':vectors[:,:,0],'valid_normals':valid}


def huber(residual):
    r=np.minimum(residual,PARAMETERS['correspondence_gate_m']);h=PARAMETERS['huber_m']
    return np.where(r<=h,.5*r*r,h*(r-.5*h))


def metrics(groups,matrix,scope='calibration'):
    rows=[]
    for group in groups:
        points=group[scope];error=group['target']['tree'].query(apply(points,matrix))[0]
        rows.append({'family':group['family'],'count':len(points),'median_m':float(np.median(error)),
            'p90_m':float(np.quantile(error,.9)),'loss':float(np.mean(huber(error)))})
    return {'loss':float(np.mean([r['loss'] for r in rows])),'families':rows}


def objective(groups,matrix,tangent_weight):
    losses=[]
    for group in groups:
        p=apply(group['fit'],matrix);t=group['target'];distance,near=t['tree'].query(p)
        residual=p-t['points'][near];normal=t['normals'][near];plane=np.sum(residual*normal,axis=1)
        norm=np.sqrt(tangent_weight*np.sum(residual**2,axis=1)+(1-tangent_weight)*plane**2)
        norm[(distance>=PARAMETERS['correspondence_gate_m'])|~t['valid_normals'][near]]=PARAMETERS['correspondence_gate_m']
        losses.append(float(np.mean(huber(norm))))
    return float(np.mean(losses))


def solve_step(jacobian,residual,weights):
    a=jacobian*np.sqrt(weights)[:,None];b=-residual*np.sqrt(weights)
    u,s,vh=np.linalg.svd(a,full_matrices=False);keep=s>s[0]*PARAMETERS['svd_relative_cutoff']
    step=vh[keep].T@((u[:,keep].T@b)/s[keep])
    return step,s


def refine(groups,initial,tangent_weight,anchors):
    matrix=initial.copy();anchor=np.mean([apply(g['fit'],initial).mean(0) for g in groups],axis=0)
    origin_anchors=apply(anchors,initial);loss=objective(groups,matrix,tangent_weight);history=[loss];singular=None
    for iteration in range(PARAMETERS['maximum_iterations']):
        jacobians=[];residuals=[];weights=[]
        for group in groups:
            t=group['target'];p=apply(group['fit'],matrix);distance,near=t['tree'].query(p)
            keep=(distance<PARAMETERS['correspondence_gate_m'])&t['valid_normals'][near]
            q=t['points'][near[keep]];n=t['normals'][near[keep]];p=p[keep]
            if len(p)<PARAMETERS['minimum_fit_points']:return initial.copy(),{'status':'insufficient_correspondences','history':history}
            projection=np.sqrt(tangent_weight)*np.eye(3)+(1-np.sqrt(tangent_weight))*n[:,:,None]*n[:,None,:]
            jac=np.concatenate([np.cross((p-anchor)[:,None,:],np.eye(3)[None,:,:])/PARAMETERS['rotation_lever_m'],np.broadcast_to(np.eye(3),(len(p),3,3))],axis=2)
            jac=np.einsum('nij,njk->nik',projection,jac);r=np.einsum('nij,nj->ni',projection,p-q)
            w=np.minimum(1,PARAMETERS['huber_m']/np.maximum(np.linalg.norm(r,axis=1),1e-12))/(len(groups)*len(group['fit']))
            jacobians.append(jac.reshape(-1,6));residuals.append(r.ravel());weights.append(np.repeat(w,3))
        step,singular=solve_step(np.concatenate(jacobians),np.concatenate(residuals),np.concatenate(weights))
        accepted=False
        for fraction in [1.,.5,.25,.125,.0625]:
            delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(step[:3]*fraction/PARAMETERS['rotation_lever_m']).as_matrix()
            delta[:3,3]=anchor-delta[:3,:3]@anchor+step[3:]*fraction;trial=delta@matrix
            motion=np.max(np.linalg.norm(apply(anchors,trial)-origin_anchors,axis=1))
            trial_loss=objective(groups,trial,tangent_weight)
            if motion<=PARAMETERS['maximum_motion_m'] and trial_loss<loss-1e-12:
                matrix=trial;loss=trial_loss;history.append(loss);accepted=True;break
        if not accepted:break
        if np.linalg.norm(step*fraction)<1e-6:break
    return matrix,{'status':'refined','iterations':iteration+1,'history':history,'tangent_weight':tangent_weight,
        'singular_values':singular.tolist(),'maximum_anchor_motion_m':float(np.max(np.linalg.norm(apply(anchors,matrix)-origin_anchors,axis=1)))}


def select_candidate(candidates,baseline):
    """Selection uses training/calibration evidence only, never final controls."""
    valid=[]
    for i,row in enumerate(candidates):
        good=row['strict_pass'] and row['global_training_guard']
        good&=row['calibration']['loss']<baseline['loss']-PARAMETERS['minimum_calibration_loss_improvement']
        good&=all(a['median_m']<=b['median_m']+PARAMETERS['maximum_family_median_regression_m'] for a,b in zip(row['calibration']['families'],baseline['families']))
        if good:valid.append(i)
    return min(valid,key=lambda i:(candidates[i]['calibration']['loss'],i)) if valid else None


def extent_anchors(points):
    lo,hi=np.quantile(points,[.01,.99],axis=0)
    return np.array(list(itertools.product(*zip(lo,hi))))
