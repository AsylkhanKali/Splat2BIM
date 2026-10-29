"""Structural refinement anchored by static-filtered whole-scene geometry."""
import numpy as np
from scipy.spatial.transform import Rotation

from align_0929 import apply
from structural_pose_refinement import PARAMETERS,huber,solve_step

GUARD_PARAMETERS={'context_voxel_m':.5,'maximum_context_points':8000,'structure_tangent_weight':.1,
    'context_total_weight':.5,'fractions':[1.,.5,.25,.125,.0625,.03125],
    'global_training_median_and_p90_tolerance_m':1e-9,'maximum_iterations':60}


def context_indices(points):
    _,ids=np.unique(np.floor(points/GUARD_PARAMETERS['context_voxel_m']).astype(int),axis=0,return_index=True)
    if len(ids)>GUARD_PARAMETERS['maximum_context_points']:
        ids=ids[np.linspace(0,len(ids)-1,GUARD_PARAMETERS['maximum_context_points'],dtype=int)]
    return np.sort(ids)


def projected(group,matrix):
    p=apply(group['fit'],matrix);t=group['target'];distance,near=t['tree'].query(p)
    q=t['points'][near];residual=p-q
    if group.get('context',False):
        projection=np.broadcast_to(np.eye(3),(len(p),3,3));valid=distance<PARAMETERS['correspondence_gate_m']
    else:
        n=t['normals'][near];root=np.sqrt(GUARD_PARAMETERS['structure_tangent_weight'])
        projection=root*np.eye(3)+(1-root)*n[:,:,None]*n[:,None,:]
        valid=(distance<PARAMETERS['correspondence_gate_m'])&t['valid_normals'][near]
    return p,np.einsum('nij,nj->ni',projection,residual),projection,valid


def objective(groups,matrix):
    total=0.
    for group in groups:
        _,residual,_,valid=projected(group,matrix);norm=np.linalg.norm(residual,axis=1)
        norm[~valid]=PARAMETERS['correspondence_gate_m'];total+=group['weight']*float(np.mean(huber(norm)))
    return total


def refine_with_context(structures,context,initial,anchors):
    context_weight=GUARD_PARAMETERS['context_total_weight']
    groups=[{**g,'weight':(1-context_weight)/len(structures)} for g in structures]
    groups.append({**context,'weight':context_weight,'context':True})
    anchor=apply(anchors,initial).mean(0);origin_anchors=apply(anchors,initial);matrix=initial.copy()
    loss=objective(groups,matrix);history=[loss];singular=None
    for iteration in range(GUARD_PARAMETERS['maximum_iterations']):
        aa=[];rr=[];ww=[]
        for group in groups:
            p,r,projection,valid=projected(group,matrix);p,r,projection=p[valid],r[valid],projection[valid]
            if len(p)<PARAMETERS['minimum_fit_points']:return initial.copy(),{'status':'insufficient_support','history':history}
            jac=np.concatenate([np.cross((p-anchor)[:,None,:],np.eye(3)[None,:,:])/PARAMETERS['rotation_lever_m'],np.broadcast_to(np.eye(3),(len(p),3,3))],axis=2)
            jac=np.einsum('nij,njk->nik',projection,jac)
            weight=np.minimum(1,PARAMETERS['huber_m']/np.maximum(np.linalg.norm(r,axis=1),1e-12))*group['weight']/len(group['fit'])
            aa.append(jac.reshape(-1,6));rr.append(r.ravel());ww.append(np.repeat(weight,3))
        step,singular=solve_step(np.concatenate(aa),np.concatenate(rr),np.concatenate(ww));accepted=False
        for fraction in [1.,.5,.25,.125,.0625]:
            delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(step[:3]*fraction/PARAMETERS['rotation_lever_m']).as_matrix()
            delta[:3,3]=anchor-delta[:3,:3]@anchor+step[3:]*fraction;trial=delta@matrix
            motion=np.linalg.norm(apply(anchors,trial)-origin_anchors,axis=1).max();value=objective(groups,trial)
            if motion<=PARAMETERS['maximum_motion_m'] and value<loss-1e-12:
                matrix=trial;loss=value;history.append(loss);accepted=True;break
        if not accepted or np.linalg.norm(step*fraction)<1e-6:break
    return matrix,{'status':'refined','history':history,'iterations':iteration+1,'singular_values':singular.tolist()}


def fractional_correction(initial,proposal,fraction,anchor):
    delta=proposal@np.linalg.inv(initial);rotation=Rotation.from_matrix(delta[:3,:3])
    result=np.eye(4);result[:3,:3]=Rotation.from_rotvec(rotation.as_rotvec()*fraction).as_matrix()
    displacement=apply(anchor[None],delta)[0]-anchor
    result[:3,3]=anchor-result[:3,:3]@anchor+fraction*displacement
    return result@initial


def global_guard(before,after):
    return all(after[k]<=before[k]+GUARD_PARAMETERS['global_training_median_and_p90_tolerance_m'] for k in ['median','p90'])
