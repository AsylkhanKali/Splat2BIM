"""Building specialist plus drone classes; all fusion rules are date independent."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

FUSION_PARAMETERS={'building_confidence':.6,'protected_drone_classes':[4,5,6],
    'source_context_radius':'max(24, ceil(0.4 / pixel_size_source_units))',
    'held_tile_width_source_units':.4,'median_kernel':3}


def expanded_context(image,points,metadata):
    """Fill image context from visible training samples, never create 3D samples."""
    size=image.shape[0];local=(points-metadata['center'])@np.array(metadata['frame'])
    uv=np.rint((local[:,:2]-metadata['origin_xy'])/metadata['pixel_size_source_units']).astype(int)
    uv[:,1]=size-1-uv[:,1];observed=np.zeros((size,size),bool);observed[uv[:,1],uv[:,0]]=True
    distance,nearest=distance_transform_edt(~observed,return_indices=True)
    radius=max(24,int(np.ceil(FUSION_PARAMETERS['held_tile_width_source_units']/metadata['pixel_size_source_units'])))
    out=np.full_like(image,128);out[observed]=image[observed]
    fill=(~observed)&(distance<=radius);out[fill]=out[nearest[0][fill],nearest[1][fill]]
    return cv2.medianBlur(out,FUSION_PARAMETERS['median_kernel']),uv,radius


def fuse(drone,expert):
    labels=drone['labels'].copy();confidence=drone['confidence'].copy();origin=np.zeros(len(labels),np.uint8)
    stale=labels==1;labels[stale]=0;confidence[stale]=0;origin[stale]=2
    built=(expert[:,1]>=FUSION_PARAMETERS['building_confidence'])&~np.isin(drone['labels'],FUSION_PARAMETERS['protected_drone_classes'])
    labels[built]=1;confidence[built]=expert[built,1];origin[built]=1
    return labels,confidence,origin
