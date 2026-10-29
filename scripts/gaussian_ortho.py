"""Training-only orthographic 3DGS views, including covariance, alpha and degree-3 SH."""
import ctypes
import hashlib
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import expit

from align_dates import ROOT
from export_3dgs import sh_basis

RENDER_PARAMETERS={'minimum_alpha':1/255,'pixel_covariance_padding':.3,'ellipse_sigma_radius':3.,
    'background_rgb':[.5,.5,.5],'source_record_stride':2,'source_tile_width':.4}


def doubled_metadata(metadata):
    """Double resolution while preserving the original outer pixel rectangle."""
    return {**metadata,'origin_xy':(np.array(metadata['origin_xy'])-.25*metadata['pixel_size_source_units']).tolist(),
        'pixel_size_source_units':metadata['pixel_size_source_units']/2}


def training_context(vertices,training):
    """Even PLY records only, within training bounds and outside withheld tiles."""
    ids=np.arange(0,len(vertices),RENDER_PARAMETERS['source_record_stride']);v=vertices[ids]
    points=np.c_[v['x'],v['y'],v['z']].astype(float)
    keep=(v['opacity']>np.log(1/254))&np.all((points>=training.min(0))&(points<=training.max(0)),axis=1)
    keep&=np.floor(points/RENDER_PARAMETERS['source_tile_width']).sum(1).astype(int)%4!=0
    return v[keep],ids[keep]


def project_gaussians(vertices,metadata,size=1024):
    points=np.c_[vertices['x'],vertices['y'],vertices['z']].astype(float)
    frame=np.array(metadata['frame']);step=metadata['pixel_size_source_units'];local=(points-metadata['center'])@frame
    uv=(local[:,:2]-metadata['origin_xy'])/step;uv[:,1]=size-1-uv[:,1]
    screen=frame[:,:2].copy();screen[:,1]*=-1
    quat=np.c_[[vertices[f'rot_{i}'] for i in range(4)]].T
    rotation=Rotation.from_quat(quat,scalar_first=True).as_matrix()
    sigma=np.exp(np.c_[[vertices[f'scale_{i}'] for i in range(3)]].T.astype(float))
    projected=np.einsum('ij,njk->nik',screen.T,rotation)*sigma[:,None,:]/step
    covariance=projected@np.swapaxes(projected,1,2);covariance[:,[0,1],[0,1]]+=RENDER_PARAMETERS['pixel_covariance_padding']
    a,b,c=covariance[:,0,0],covariance[:,0,1],covariance[:,1,1];det=a*c-b*b
    direction=-frame[:,2][None];basis=np.concatenate([sh_basis(direction,k)[0] for k in [1,2,3]])
    dc=np.c_[[vertices[f'f_dc_{i}'] for i in range(3)]].T
    rest=np.c_[[vertices[f'f_rest_{i}'] for i in range(45)]].T.reshape(-1,3,15)
    color=np.clip(.5+.28209479177387814*dc+rest@basis,0,1)
    rows=np.c_[uv,c/det,-b/det,a/det,3*np.sqrt(a),3*np.sqrt(c),expit(vertices['opacity']),color]
    assert np.isfinite(rows).all()
    return np.ascontiguousarray(rows[np.argsort(local[:,2],kind='stable')],dtype=np.float32)


def render(rows,size=1024):
    source=ROOT/'scripts/gaussian_ortho.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=ROOT/'.cache';folder.mkdir(exist_ok=True);library=folder/f'gaussian_ortho_{digest}.dylib'
    if not library.exists():subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library));pointer=ctypes.POINTER(ctypes.c_float)
    lib.render_gaussians.argtypes=[pointer,ctypes.c_int,ctypes.c_int,pointer,pointer];lib.render_gaussians.restype=None
    rows=np.ascontiguousarray(rows,dtype=np.float32);rgb=np.full((size,size,3),.5,np.float32);alpha=np.zeros((size,size),np.float32)
    lib.render_gaussians(rows.ctypes.data_as(pointer),len(rows),size,rgb.ctypes.data_as(pointer),alpha.ctypes.data_as(pointer))
    return np.uint8(np.clip(rgb,0,1)*255),alpha
