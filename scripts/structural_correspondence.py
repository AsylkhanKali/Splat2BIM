"""Alternative surface extents with one support per source/target surface family."""
import itertools

import numpy as np
from scipy.spatial.distance import cdist

from align_0929 import apply
from object_consensus import PARAMETERS


def combine_surfaces(original,grown):
    parent={r['id']:r['id'] for r in original['candidates']}
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    for row in grown['candidates']:
        roots=sorted({root(s) for s in row['seed_ids']})
        for name in roots:parent[name]=roots[0]
    rows=[]
    for kind,found in [('original',original),('full',grown)]:
        for row in found['candidates']:
            seeds=row.get('seed_ids',[row['id']]);rows.append({**row,'id':kind+'_'+row['id'],'seed_ids':seeds,
                'family':root(seeds[0]),'extent_kind':kind})
    return {**original,'candidates':rows}


def gated_pairs(source,target,matrix):
    """All identities inside unchanged gates, before any one-to-one decision."""
    if not source['ids'] or not target['ids']:return []
    p=PARAMETERS;scale=np.cbrt(np.linalg.det(matrix[:3,:3]));rotation=matrix[:3,:3]/scale
    distance=cdist(apply(source['centers'],matrix),target['centers'])
    ratio=source['lengths'][:,None,:]*scale/target['lengths'][None,:,:]
    normal=(source['normals']@rotation.T)@target['normals'].T
    valid=(distance<=p['center_gate_m'])&np.all((ratio>=p['dimension_ratio_min'])&(ratio<=p['dimension_ratio_max']),axis=2)
    valid&=normal>=np.cos(np.deg2rad(p['normal_angle_degrees']))
    return [{'source_index':int(i),'target_index':int(j),'source_id':source['ids'][i],'target_id':target['ids'][j],
        'center_distance_m':float(distance[i,j])} for i,j in zip(*np.where(valid))]


def family_support(pairs,sf,tf,matrix):
    """Find a valid triangle, then extend it without duplicate physical families."""
    p=PARAMETERS
    source=np.array([r['center'] for r in sf['candidates']])/sf['normalization_scale']+sf['normalization_center']
    target=np.array([r['center'] for r in tf['candidates']])/tf['normalization_scale']+tf['normalization_center'];source=apply(source,matrix)
    family=lambda pair:(sf['candidates'][pair['source_index']]['family'],tf['candidates'][pair['target_index']]['family'])
    ranked=sorted(pairs,key=lambda r:(-min(r.get('fraction_within_1m',1),r.get('reverse_fraction_within_1m',1)),r['center_distance_m'],r['source_id'],r['target_id']))
    compatible=lambda a,b:all(x!=y for x,y in zip(family(a),family(b))) and np.linalg.norm(source[a['source_index']]-source[b['source_index']])>=p['min_separation_m'] and np.linalg.norm(target[a['target_index']]-target[b['target_index']])>=p['min_separation_m']
    chosen=None
    for triple in itertools.combinations(ranked,3):
        if not all(compatible(a,b) for a,b in itertools.combinations(triple,2)):continue
        if all(np.linalg.svd(points[[r[key] for r in triple]]-points[[r[key] for r in triple]].mean(0),compute_uv=False)[1]>=p['min_second_axis_m'] for points,key in [(source,'source_index'),(target,'target_index')]):
            chosen=list(triple);break
    if chosen is None:return []
    for pair in ranked:
        if all(compatible(pair,old) for old in chosen):chosen.append(pair)
    return chosen


def initial_family_bank(matrices,index,anchors,source,target):
    """Keep identities proposed by nearby initial hypotheses; never by final poses."""
    moved=np.einsum('nj,kij->kni',anchors,matrices[:,:3,:3])+matrices[:,None,:3,3]
    rms=np.sqrt(np.mean(np.sum((moved-moved[index])**2,axis=2),axis=1));neighbors=np.flatnonzero(rms<2.)
    bank={}
    for neighbor in neighbors:
        for pair in gated_pairs(source,target,matrices[neighbor]):
            key=(pair['source_id'],pair['target_id'])
            if key not in bank:bank[key]={**pair,'proposed_by_pool_indices':[]}
            bank[key]['proposed_by_pool_indices'].append(int(neighbor))
    return [bank[k] for k in sorted(bank)],neighbors.tolist()
