"""Rebuild the frozen 0929 semantic inputs from source assets and pinned models."""
import argparse,json
from pathlib import Path
import cv2,laspy
import numpy as np
from scipy.ndimage import distance_transform_edt
from plyfile import PlyData
from project_data import ROOT,ORIGIN,file_hash
from semantic_projection import Segmenter,rasterize,visible_labels,COLORS
from building_segmenter import StructureSegmenter
from semantic_fusion import fuse
from gaussian_ortho import doubled_metadata,training_context,project_gaussians,render


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out
    if out.exists():raise FileExistsError('Use a fresh output directory to preserve frozen semantic inputs.')
    out.mkdir(parents=True)
    model_inputs={}
    for model in ['drone_segformer_b0','loveda_segformer_b2']:
        folder=ROOT/'models'/model;record=json.loads((folder/'provenance.json').read_text())
        for name,digest in record['files'].items():assert file_hash(folder/name)==digest,(model,name)
        model_inputs[model]=record
    vertices=PlyData.read(ROOT/'assets/0929/0929.ply',mmap='r')['vertex'].data
    v=vertices[::10];bench=np.c_[v['x'],v['y'],v['z']].astype(float)
    bench=bench[np.isfinite(bench).all(1)&(v['opacity']>0)]
    lo,hi=np.quantile(bench,[.005,.995],axis=0);bench=bench[np.all((bench>=lo)&(bench<=hi),axis=1)]
    ids=np.arange(0,len(vertices),2);v=vertices[ids];points=np.c_[v['x'],v['y'],v['z']].astype(float)
    keep=(v['opacity']>0)&np.isfinite(points).all(1)&np.all((points>=bench.min(0))&(points<=bench.max(0)),axis=1)
    keep&=np.floor(points/.4).sum(1).astype(int)%4!=0
    source=points[keep];ids=ids[keep]
    rgb=np.clip(.5+.28209479177387814*np.c_[v['f_dc_0'],v['f_dc_1'],v['f_dc_2']],0,1)[keep]
    las=laspy.read(ROOT/'assets/0929/0929.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
    projection=json.loads((ROOT/'results/segmentation/projection.json').read_text())
    drone_model=Segmenter(ROOT/'models/drone_segformer_b0');building=StructureSegmenter(ROOT/'models/loveda_segformer_b2')
    for side,p,c in [('source',source,rgb),('target',target,color)]:
        image,index,meta=rasterize(p,c,projection[side]['up'],1024)
        probabilities=drone_model.predict(image);drone=visible_labels(p,index,probabilities)
        if side=='source':
            drone['original_ply_indices']=ids[drone['input_indices']]
            meta=doubled_metadata(meta);context,context_ids=training_context(vertices,source)
            image,alpha=render(project_gaussians(context,meta,2048),2048)
            local=(drone['points']-meta['center'])@np.array(meta['frame'])
            uv=np.rint((local[:,:2]-meta['origin_xy'])/meta['pixel_size_source_units']).astype(int);uv[:,1]=2047-uv[:,1]
            expert=building.predict(image)[uv[:,1],uv[:,0]]
        else:
            observed=index>=0;distance,near=distance_transform_edt(~observed,return_indices=True)
            fill=(~observed)&(distance<=24);image[fill]=image[near[0][fill],near[1][fill]];image=cv2.medianBlur(image,3)
            expert=building.predict(image)[observed].astype(np.float16).astype(np.float32)
            yy,xx=np.nonzero(observed);uv=np.c_[xx,yy]
        labels,confidence,origin=fuse(drone,expert)
        data={**drone,'labels':labels,'confidence':confidence,'label_origin':origin,'building_expert_probabilities':expert if side=='source' else expert.astype(np.float16)}
        if side=='source':data['render_uv']=uv
        np.savez_compressed(out/f'{side}_labels.npz',**data)
        semantic=np.full_like(image,245);semantic[uv[:,1],uv[:,0]]=COLORS[labels]
        for suffix,img in [('rgb',image),('semantic',semantic)]:cv2.imwrite(str(out/f'{side}_{suffix}.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        print(side,len(labels),dict(zip(*np.unique(labels,return_counts=True))),flush=True)
    (out/'provenance.json').write_text(json.dumps({'date':'0929','models':model_inputs,'projection_sha256':file_hash(ROOT/'results/segmentation/projection.json'),
        'label_sha256':{side:file_hash(out/f'{side}_labels.npz') for side in ['source','target']},
        'note':'Reproduces historical development settings, including globally estimated cropping bounds; no fitted alignment matrix is read. Held-out centers are excluded; Gaussian footprints may cross tile boundaries.'},indent=2)+'\n')

if __name__=='__main__':main()
