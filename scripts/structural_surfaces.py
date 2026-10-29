"""Grow connected, full-color planar surfaces independently in each cloud."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from object_candidates import describe_cluster
from object_consensus import PARAMETERS as GATES

PARAMETERS={'voxel':.12,'plane_distance':.05,'cluster_eps':.65,'cluster_min_points':8,
    'minimum_points':40,'semantic_exclusion_distance':.24,'excluded_classes':[4,5,6],
    'seed_attachment_distance':.24,'deduplicate_overlap':.5,'minimum_persistent_dates':3}


def grow_surfaces(points,labels,found):
    p=PARAMETERS;center=np.array(found['normalization_center']);scale=found['normalization_scale']
    normalized=(points-center)*scale
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normalized));cloud=cloud.voxel_down_sample(p['voxel'])
    xyz=np.asarray(cloud.points).copy()
    distance,nearest=cKDTree((labels['points']-center)*scale).query(xyz)
    excluded=(distance<=p['semantic_exclusion_distance'])&np.isin(labels['labels'][nearest],p['excluded_classes'])
    xyz=xyz[~excluded];groups={};proposals=[]
    for seed in found['candidates']:groups.setdefault(seed['id'].split('_')[0],[]).append(seed)
    for plane_id,seeds in sorted(groups.items()):
        seed_points=np.concatenate([s['points'] for s in seeds]);origin=seed_points.mean(0)
        _,_,vh=np.linalg.svd(seed_points-origin,full_matrices=False);normal=vh[-1]
        ids=np.flatnonzero(abs((xyz-origin)@normal)<=p['plane_distance'])
        if len(ids)<p['minimum_points']:continue
        patch=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[ids]))
        components=np.asarray(patch.cluster_dbscan(eps=p['cluster_eps'],min_points=p['cluster_min_points']))
        tree=cKDTree(xyz[ids]);attached={}
        for seed in seeds:
            distance,near=tree.query(seed['points']);hits=components[near[distance<=p['seed_attachment_distance']]]
            for component in np.unique(hits[hits>=0]):attached.setdefault(int(component),[]).append(seed['id'])
        for component,seed_ids in sorted(attached.items()):
            selected=ids[components==component]
            if len(selected)<p['minimum_points']:continue
            row=describe_cluster(xyz[selected],np.array(found['ground_plane_normalized']))
            if not (row['height']>=GATES['min_height'] and row['rectangle_fill']>=GATES['min_rectangle_fill'] and
                row['lengths'][0]/row['lengths'][1]<=GATES['max_aspect'] and row['lengths'][1]>=GATES['min_width'] and
                row['height']/row['lengths'][0]<=GATES['max_height_to_length'] and row['ground_angle_degrees']<=GATES['max_ground_angle_degrees']):continue
            row.update({'id':f'{plane_id}_full{component}','seed_ids':sorted(seed_ids),'points':xyz[selected],
                'voxel_indices':selected,'growth_plane_origin':origin,'growth_plane_normal':normal})
            proposals.append(row)
    # Overlapping proposals from different RANSAC slices cannot count twice.
    selected=[]
    for row in sorted(proposals,key=lambda r:(-len(r['points']),r['id'])):
        duplicate=next((old for old in selected if len(np.intersect1d(row['voxel_indices'],old['voxel_indices']))/min(len(row['points']),len(old['points']))>=p['deduplicate_overlap']),None)
        if duplicate is None:selected.append(row)
        else:duplicate['seed_ids']=sorted(set(duplicate['seed_ids']+row['seed_ids']))
    selected.sort(key=lambda r:r['id'])
    return {**{k:found[k] for k in ['normalization_center','normalization_scale','ground_plane_normalized']},
        'candidates':selected,'input_count':len(points),'voxel_count':len(xyz),'excluded_voxel_count':int(excluded.sum()),
        'original_seed_count':len(found['candidates']),'before_deduplication':len(proposals)}


def serializable(found):
    return {k:[{name:(value.tolist() if isinstance(value,np.ndarray) else value) for name,value in row.items() if name not in ['points','voxel_indices']} for row in v]
        if k=='candidates' else v.tolist() if isinstance(v,np.ndarray) else v for k,v in found.items()}
