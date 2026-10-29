"""Fuse cached drone labels with LoveDA building predictions from training context."""
import argparse
import json
import time

import cv2
import numpy as np

from align_dates import ROOT,DATES
from check_las_vertical import file_hash
from semantic_projection import GROUPS,COLORS
from semantic_structures import StructureSegmenter
from semantic_fusion import FUSION_PARAMETERS,expanded_context,fuse
from prepare_semantic_clouds import OUT as OLD
from prepare_semantic_structures import OUT as EXPERT,MODEL

OUT=ROOT/'results/semantic_fusion'


def main(dates):
    OUT.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((MODEL/'provenance.json').read_text())
    for name,digest in provenance['files'].items():assert file_hash(MODEL/name)==digest
    protocol={'parameters':FUSION_PARAMETERS,'model':provenance,
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['semantic_fusion.py','semantic_structures.py','prepare_semantic_fusion.py']},
        'old_protocol_sha256':file_hash(OLD/'segmentation_protocol.json'),'expert_protocol_sha256':file_hash(EXPERT/'segmentation_protocol.json'),
        'alignment_matrices_used_for_predictions':False,'held_out_points_used_for_predictions':False}
    path=OUT/'segmentation_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    segmenter=StructureSegmenter(MODEL)
    for date in dates:
        folder=OUT/date;folder.mkdir(exist_ok=True)
        if (folder/'segmentation.json').exists():continue
        start=time.monotonic();metadata=json.loads((OLD/date/'segmentation.json').read_text());records={};inputs={}
        for side in ['source','target']:
            input_path=OLD/date/f'{side}_labels.npz';drone=dict(np.load(input_path));expert_path=EXPERT/date/f'{side}_labels.npz';expert=dict(np.load(expert_path))
            np.testing.assert_array_equal(drone['input_indices'],expert['input_indices']);np.testing.assert_array_equal(drone['points'],expert['points'])
            rgb_path=OLD/date/f'{side}_rgb.png';image=cv2.cvtColor(cv2.imread(str(rgb_path)),cv2.COLOR_BGR2RGB)
            m=metadata['clouds'][side]
            if side=='source':
                image,uv,radius=expanded_context(image,drone['points'],m)
                probability=segmenter.predict(image)[uv[:,1],uv[:,0]]
            else:
                local=(drone['points']-m['center'])@np.array(m['frame']);uv=np.rint((local[:,:2]-m['origin_xy'])/m['pixel_size_source_units']).astype(int);uv[:,1]=1023-uv[:,1]
                radius=24;probability=expert['probabilities'].astype(np.float32)
                image=cv2.cvtColor(cv2.imread(str(EXPERT/date/f'{side}_rgb.png')),cv2.COLOR_BGR2RGB)
            labels,confidence,origin=fuse(drone,probability)
            data={**drone,'labels':labels,'confidence':confidence,'label_origin':origin,'building_expert_probabilities':probability.astype(np.float16)}
            np.savez_compressed(folder/f'{side}_labels.npz',**data)
            color=np.full_like(image,245);color[uv[:,1],uv[:,0]]=COLORS[labels]
            cv2.imwrite(str(folder/f'{side}_rgb.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));cv2.imwrite(str(folder/f'{side}_semantic.png'),cv2.cvtColor(color,cv2.COLOR_RGB2BGR))
            counts=dict(zip(GROUPS,np.bincount(labels,minlength=len(GROUPS)).tolist()))
            records[side]={**m,'class_counts':counts,'context_radius_pixels':radius,'data_sha256':file_hash(folder/f'{side}_labels.npz')}
            for p in [input_path,expert_path,rgb_path,OLD/date/'segmentation.json']:inputs[str(p.relative_to(ROOT))]=file_hash(p)
            print(date,side,counts,flush=True)
        (folder/'segmentation.json').write_text(json.dumps({'date':date,'clouds':records,'inputs':inputs,'seconds':time.monotonic()-start},indent=2)+'\n')
        print('Finished fusion',date,round(time.monotonic()-start,1),'s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='*',default=DATES);main(p.parse_args().dates)
