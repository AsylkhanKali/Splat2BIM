"""Classify real orthographic Gaussian views with the existing building expert."""
import argparse
import json
import subprocess
import time
from importlib.metadata import version

import cv2
import numpy as np
from plyfile import PlyData

from align_dates import ROOT,DATES,samples
from align_objects import dense_training
from check_las_vertical import file_hash
from semantic_projection import GROUPS,COLORS
from semantic_structures import StructureSegmenter
from semantic_fusion import fuse,FUSION_PARAMETERS
from gaussian_ortho import doubled_metadata,training_context,project_gaussians,render,RENDER_PARAMETERS
from prepare_semantic_clouds import OUT as OLD
from prepare_semantic_fusion import OUT as PREVIOUS
from prepare_semantic_structures import MODEL

OUT=ROOT/'results/semantic_3dgs'
SIZE=2048


def main(dates):
    OUT.mkdir(exist_ok=True)
    provenance=json.loads((MODEL/'provenance.json').read_text())
    for name,digest in provenance['files'].items():assert file_hash(MODEL/name)==digest
    protocol={'model':provenance,'render':RENDER_PARAMETERS,'raster_size':SIZE,
        'fusion':{k:FUSION_PARAMETERS[k] for k in ['building_confidence','protected_drone_classes']},
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['gaussian_ortho.cpp','gaussian_ortho.py','prepare_semantic_3dgs.py','semantic_structures.py','semantic_fusion.py']},
        'runtime':{n:version(n) for n in ['torch','transformers','numpy','scipy','plyfile','opencv-python-headless']},
        'compiler':subprocess.check_output(['clang++','--version'],text=True).splitlines()[0],
        'previous_protocol_sha256':file_hash(PREVIOUS/'segmentation_protocol.json'),
        'alignment_matrices_used_for_rendering_or_prediction':False,'held_out_centers_used_for_prediction':False,
        'footprints_may_overlap_excluded_tiles':True,'unchanged_target_labels':True,'registration_rerun':False,
        'development_note':'Resolution selected after exploratory roof checks on 0723 and 0929; all seven dates use the same settings. These regions are development diagnostics, not an independent semantic test set.'}
    path=OUT/'segmentation_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Preserve the frozen protocol.'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    segmenter=StructureSegmenter(MODEL)
    for date in dates:
        folder=OUT/date;folder.mkdir(exist_ok=True)
        if (folder/'segmentation.json').exists():print('Cached',date,flush=True);continue
        start=time.monotonic();old_metadata=json.loads((OLD/date/'segmentation.json').read_text())
        metadata=doubled_metadata(old_metadata['clouds']['source']);drone=dict(np.load(OLD/date/'source_labels.npz'))
        benchmark,_,_=samples(date);training,_,_=dense_training(date,benchmark)
        ply_path=ROOT/f'assets/{date}/{date}.ply';vertices=PlyData.read(ply_path)['vertex'].data
        context,ids=training_context(vertices,training);np.savez_compressed(folder/'render_context.npz',original_ply_indices=ids)
        rgb,alpha=render(project_gaussians(context,metadata,SIZE),SIZE)
        print(date,'rendered',len(ids),'Gaussians',round(time.monotonic()-start,1),'s',flush=True)
        cv2.imwrite(str(folder/'source_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(folder/'source_alpha.png'),np.uint8(alpha*255))
        local=(drone['points']-metadata['center'])@np.array(metadata['frame'])
        uv=np.rint((local[:,:2]-metadata['origin_xy'])/metadata['pixel_size_source_units']).astype(int);uv[:,1]=SIZE-1-uv[:,1]
        assert np.all((uv>=0)&(uv<SIZE))
        probability=segmenter.predict(rgb)[uv[:,1],uv[:,0]]
        labels,confidence,origin=fuse(drone,probability)
        np.savez_compressed(folder/'source_labels.npz',**{**drone,'labels':labels,'confidence':confidence,
            'label_origin':origin,'building_expert_probabilities':probability,'render_uv':uv})
        colors=np.full_like(rgb,245);colors[uv[:,1],uv[:,0]]=COLORS[labels]
        cv2.imwrite(str(folder/'source_semantic.png'),cv2.cvtColor(colors,cv2.COLOR_RGB2BGR))
        overlay=rgb.copy();built=labels==1;overlay[uv[built,1],uv[built,0]]=COLORS[1]
        cv2.imwrite(str(folder/'source_built.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
        inputs=[ply_path,OLD/date/'source_labels.npz',OLD/date/'segmentation.json',PREVIOUS/date/'source_labels.npz',PREVIOUS/date/'target_labels.npz']
        record={'date':date,'clouds':{'source':{**metadata,'raster_size':SIZE,
            'class_counts':dict(zip(GROUPS,np.bincount(labels,minlength=len(GROUPS)).tolist())),
            'data_sha256':file_hash(folder/'source_labels.npz')}},
            'target_labels_reference':str((PREVIOUS/date/'target_labels.npz').relative_to(ROOT)),
            'render_context_count':len(ids),'render_context_bounds':[training.min(0).tolist(),training.max(0).tolist()],
            'render_context_sha256':file_hash(folder/'render_context.npz'),
            'image_sha256':file_hash(folder/'source_rgb.png'),
            'inputs':{str(p.relative_to(ROOT)):file_hash(p) for p in inputs},'seconds':time.monotonic()-start}
        assert record['inputs'][str(ply_path.relative_to(ROOT))]==old_metadata['inputs'][str(ply_path.relative_to(ROOT))]
        (folder/'segmentation.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Finished',date,record['clouds']['source']['class_counts'],round(time.monotonic()-start,1),'s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='*',default=DATES);main(p.parse_args().dates)
