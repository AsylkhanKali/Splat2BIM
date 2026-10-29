"""Spatial agreement between surface fragments; no semantic class prediction."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from align_0929 import apply,similarity
from object_candidates import detect,guesses


PARAMETERS={'min_height':.2,'min_rectangle_fill':.5,'max_aspect':6.,
    'min_width':.6,'max_height_to_length':.75,'max_ground_angle_degrees':15.,
    'center_gate_m':5.,'dimension_ratio_min':.5,'dimension_ratio_max':2.,
    'normal_angle_degrees':20.,'min_pairs':3,'min_separation_m':10.,
    'min_second_axis_m':2.,'min_patch_fraction_each_direction':.5}


def fragments(points,colors):
    found=detect(points,colors);p=PARAMETERS
    found['candidates']=[c for c in found['clusters'] if c['height']>=p['min_height'] and
        c['rectangle_fill']>=p['min_rectangle_fill'] and c['lengths'][0]/c['lengths'][1]<=p['max_aspect'] and
        c['lengths'][1]>=p['min_width'] and c['height']/c['lengths'][0]<=p['max_height_to_length'] and
        c['ground_angle_degrees']<=p['max_ground_angle_degrees']]
    return found


def descriptors(found):
    rows=found['candidates'];scale=found['normalization_scale']
    return {'ids':[r['id'] for r in rows],
        'centers':np.array([r['center'] for r in rows]).reshape(-1,3)/scale+found['normalization_center'],
        'lengths':np.array([r['lengths'] for r in rows]).reshape(-1,2)/scale,
        'normals':np.array([np.asarray(r['frame'])[:,2] for r in rows]).reshape(-1,3)}


def match(source,target,matrix):
    """Maximum-cardinality one-to-one assignment inside fixed geometric gates."""
    if not len(source['ids']) or not len(target['ids']):return []
    p=PARAMETERS;scale=np.cbrt(np.linalg.det(matrix[:3,:3]));rotation=matrix[:3,:3]/scale
    distance=cdist(apply(source['centers'],matrix),target['centers'])
    ratio=source['lengths'][:,None,:]*scale/target['lengths'][None,:,:]
    normal=(source['normals']@rotation.T)@target['normals'].T
    valid=(distance<=p['center_gate_m'])&np.all((ratio>=p['dimension_ratio_min'])&(ratio<=p['dimension_ratio_max']),axis=2)
    valid&=normal>=np.cos(np.deg2rad(p['normal_angle_degrees']))
    rows,cols=linear_sum_assignment(np.where(valid,distance,1e6))
    return [{'source_index':int(i),'target_index':int(j),'source_id':source['ids'][i],
        'target_id':target['ids'][j],'center_distance_m':float(distance[i,j])}
        for i,j in zip(rows,cols) if valid[i,j]]


def spatial_support(pairs,source,target,matrix):
    """Repeated/nearby fragments cannot count as independent spatial support."""
    selected=[];src=[];dst=[];moved=apply(source['centers'],matrix)
    for pair in sorted(pairs,key=lambda r:(r['center_distance_m'],r['source_id'],r['target_id'])):
        a=moved[pair['source_index']];b=target['centers'][pair['target_index']]
        if selected and (np.linalg.norm(np.array(src)-a,axis=1).min()<PARAMETERS['min_separation_m'] or
            np.linalg.norm(np.array(dst)-b,axis=1).min()<PARAMETERS['min_separation_m']):continue
        selected.append(pair);src.append(a);dst.append(b)
    if len(selected)<PARAMETERS['min_pairs']:return []
    for points in (np.array(src),np.array(dst)):
        if np.linalg.svd(points-points.mean(0),compute_uv=False)[1]<PARAMETERS['min_second_axis_m']:return []
    return selected


def hypotheses(source_found,target_found):
    source=descriptors(source_found);target=descriptors(target_found)
    seeds=guesses(source_found,target_found);consensus=[]
    for index,seed in enumerate(seeds):
        matrix=seed['matrix'];initial_scale=np.cbrt(np.linalg.det(matrix[:3,:3]))
        pairs=match(source,target,matrix)
        if not spatial_support(pairs,source,target,matrix):continue
        for _ in range(3):
            fitted=similarity(source['centers'][[p['source_index'] for p in pairs]],
                target['centers'][[p['target_index'] for p in pairs]],with_scale=True)
            scale=np.cbrt(np.linalg.det(fitted[:3,:3]))
            if not .7<=scale/initial_scale<=1.4:break
            updated=match(source,target,fitted)
            if not spatial_support(updated,source,target,fitted):break
            matrix,pairs=fitted,updated
        support=spatial_support(pairs,source,target,matrix)
        consensus.append({'seed_index':index,'matrix':matrix,'pairs':pairs,'support':support})
    return seeds,consensus
