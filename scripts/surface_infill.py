"""Densify original footprint extents using measured full-color planar samples."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from object_candidates import describe_cluster
from object_consensus import PARAMETERS as GATES
from structural_surfaces import PARAMETERS


def infill_surfaces(points,labels,original):
    p=PARAMETERS;center=np.array(original['normalization_center']);scale=original['normalization_scale']
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector((points-center)*scale));cloud=cloud.voxel_down_sample(p['voxel'])
    xyz=np.asarray(cloud.points).copy();distance,nearest=cKDTree((labels['points']-center)*scale).query(xyz)
    xyz=xyz[~((distance<=p['semantic_exclusion_distance'])&np.isin(labels['labels'][nearest],p['excluded_classes']))]
    tree=cKDTree(xyz);rows=[]
    for seed in original['candidates']:
        frame=np.array(seed['frame']);origin=np.array(seed['center']);lengths=np.array(seed['lengths'])
        indices=tree.query_ball_point(origin,np.linalg.norm(lengths)/2+p['plane_distance']);q=xyz[indices]
        local=(q-origin)@frame
        mask=(abs(local[:,2])<=p['plane_distance'])&np.all(abs(local[:,:2])<=lengths/2+1e-9,axis=1);q=q[mask]
        if len(q)<p['minimum_points']:continue
        patch=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q));components=np.asarray(patch.cluster_dbscan(eps=p['cluster_eps'],min_points=p['cluster_min_points']))
        distance,near=cKDTree(q).query(seed['points']);attached=set(components[near[distance<=p['seed_attachment_distance']]])-{-1}
        q=q[np.isin(components,list(attached))]
        if len(q)<p['minimum_points']:continue
        row=describe_cluster(q,np.array(original['ground_plane_normalized']))
        if not (row['height']>=GATES['min_height'] and row['rectangle_fill']>=GATES['min_rectangle_fill'] and
            row['lengths'][0]/row['lengths'][1]<=GATES['max_aspect'] and row['lengths'][1]>=GATES['min_width'] and
            row['height']/row['lengths'][0]<=GATES['max_height_to_length'] and row['ground_angle_degrees']<=GATES['max_ground_angle_degrees']):continue
        row.update({'id':'infill_'+seed['id'],'points':q,'seed_ids':[seed['id']],'extent_kind':'infill'})
        rows.append(row)
    return rows


def add_infill(combined,rows):
    families={seed:r['family'] for r in combined['candidates'] for seed in r['seed_ids']}
    return {**combined,'candidates':combined['candidates']+[{**r,'family':families[r['seed_ids'][0]]} for r in rows]}
