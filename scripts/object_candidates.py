"""Geometry/color rules for raised rectangular surface candidates, not a learned classifier."""
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


PARAMETERS={'white_min_channel':.55,'white_max_chroma':.18,'normalized_extent':100.,
    'voxel':.12,'plane_distance':.05,'plane_iterations':1500,'planes':12,
    'cluster_eps':.65,'cluster_min_points':8,'candidate_min_points':40,
    'min_height':.4,'max_height_to_length':.75,'min_rectangle_fill':.75,
    'max_aspect':3.,'min_width':.6,'max_length':20.,'max_ground_angle_degrees':15.}


def rectangle(points):
    center=points.mean(0);_,vectors=np.linalg.eigh(np.cov(points.T));basis=vectors[:,-2:]
    xy=(points-center)@basis;hull=ConvexHull(xy);edge=np.roll(xy[hull.vertices],-1,axis=0)-xy[hull.vertices]
    best=None
    for angle in np.unique(np.mod(np.arctan2(edge[:,1],edge[:,0]),np.pi/2)):
        axes=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        uv=xy@axes;lo=uv.min(0);hi=uv.max(0);area=np.prod(hi-lo)
        if best is None or area<best[0]:best=(area,axes,lo,hi)
    area,axes,lo,hi=best;frame2=basis@axes;lengths=hi-lo
    midpoint=center+frame2@((lo+hi)/2)
    if lengths[0]<lengths[1]:lengths=lengths[::-1];frame2=frame2[:,::-1]
    frame=np.c_[frame2,np.cross(frame2[:,0],frame2[:,1])]
    return {'center':midpoint,'frame':frame,'lengths':lengths,'area':float(area),
        'rectangle_fill':float(hull.volume/area)}


def normalize(points):
    center=np.median(points,axis=0)
    scale=PARAMETERS['normalized_extent']/np.linalg.norm(np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0))
    return (points-center)*scale,center,float(scale)


def describe_cluster(points,ground):
    candidate=rectangle(points);normal=candidate['frame'][:,2]
    height=float(candidate['center']@ground[:3]+ground[3]);length,width=candidate['lengths']
    angle=np.rad2deg(np.arccos(np.clip(abs(normal@ground[:3]),0,1)))
    p=PARAMETERS
    accepted=(len(points)>=p['candidate_min_points'] and height>=p['min_height'] and
        height/length<=p['max_height_to_length'] and candidate['rectangle_fill']>=p['min_rectangle_fill'] and
        length/width<=p['max_aspect'] and width>=p['min_width'] and length<=p['max_length'] and angle<=p['max_ground_angle_degrees'])
    if normal@ground[:3]<0:candidate['frame'][:,1:]*=-1
    return {**candidate,'count':len(points),'height':height,'ground_angle_degrees':float(angle),'accepted':bool(accepted)}


def detect(points,colors,seed=29):
    normalized,center,scale=normalize(points);p=PARAMETERS
    white=(colors.min(1)>p['white_min_channel'])&(np.ptp(colors,axis=1)<p['white_max_chroma'])
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normalized[white]));cloud=cloud.voxel_down_sample(p['voxel'])
    o3d.utility.random.seed(seed);candidates=[];all_clusters=[];ground=None
    for plane_index in range(p['planes']):
        if len(cloud.points)<p['candidate_min_points']:break
        plane,ids=cloud.segment_plane(distance_threshold=p['plane_distance'],ransac_n=3,num_iterations=p['plane_iterations'])
        if len(ids)<p['candidate_min_points']:break
        if ground is None:
            ground=np.array(plane);signed=normalized@ground[:3]+ground[3]
            if np.quantile(signed,.99)<abs(np.quantile(signed,.01)):ground=-ground
        patch=cloud.select_by_index(ids);labels=np.asarray(patch.cluster_dbscan(eps=p['cluster_eps'],min_points=p['cluster_min_points']))
        xyz=np.asarray(patch.points)
        for label in np.unique(labels):
            q=xyz[labels==label]
            if label<0 or len(q)<p['candidate_min_points']:continue
            candidate=describe_cluster(q,ground);candidate['id']=f'p{plane_index}_c{label}'
            candidate['points']=q;all_clusters.append(candidate)
            if candidate['accepted']:candidates.append(candidate)
        cloud=cloud.select_by_index(ids,invert=True)
    return {'candidates':candidates,'clusters':all_clusters,'normalization_center':center,
        'normalization_scale':scale,'ground_plane_normalized':ground,'input_count':len(points),'white_count':int(white.sum())}


def guesses(source,target):
    """Match rectangle frames; use whole-scene scale to tolerate partial object coverage."""
    rows=[];scale=source['normalization_scale']/target['normalization_scale']
    for a in source['candidates']:
        for b in target['candidates']:
            ratio=np.sqrt(b['area']/a['area'])
            if not .65<=ratio<=1.35:continue
            for turn in range(4):
                angle=turn*np.pi/2;rot2=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
                rotation=b['frame']@rot2@a['frame'].T
                ca=a['center']/source['normalization_scale']+source['normalization_center']
                cb=b['center']/target['normalization_scale']+target['normalization_center']
                matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=cb-matrix[:3,:3]@ca
                rows.append({'source_id':a['id'],'target_id':b['id'],'quarter_turn':turn,'matrix':matrix})
    return rows


def serializable(detection):
    return {k:[{f:(v.tolist() if isinstance(v,np.ndarray) else v) for f,v in c.items() if f!='points'} for c in value]
        if k in ('candidates','clusters') else value.tolist() if isinstance(value,np.ndarray) else value for k,value in detection.items()}
