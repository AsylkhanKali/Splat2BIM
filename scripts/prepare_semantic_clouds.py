"""Infer semantic classes without reading fitted or working alignment matrices."""
import argparse
import json
import time
from importlib.metadata import version

import laspy
import numpy as np

from align_dates import ROOT,ORIGIN,DATES,samples
from align_objects import dense_training
from check_las_vertical import file_hash
from semantic_projection import PARAMETERS,GROUPS,Segmenter,rasterize,visible_labels,save_images

OUT=ROOT/'results/semantic_matching';MODEL=ROOT/'models/drone_segformer_b0'


def main(dates):
    OUT.mkdir(parents=True,exist_ok=True)
    model=json.loads((MODEL/'provenance.json').read_text())
    for name,digest in model['files'].items():assert file_hash(MODEL/name)==digest
    manifest={'model':model,'parameters':PARAMETERS,'groups':GROUPS,
        'runtime':{n:version(n) for n in ['torch','transformers','safetensors','numpy','opencv-python-headless']},
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['semantic_projection.py','prepare_semantic_clouds.py']},
        'query_tiles_used_for_predictions':False,'alignment_matrices_used_for_rendering':False}
    old=OUT/'segmentation_protocol.json'
    if old.exists():assert json.loads(old.read_text())==manifest,'Segmentation protocol changed; preserve the previous experiment.'
    else:old.write_text(json.dumps(manifest,indent=2)+'\n')
    segmenter=Segmenter(MODEL)
    for date in dates:
        begin=time.monotonic();folder=OUT/date;folder.mkdir(exist_ok=True)
        if (folder/'segmentation.json').exists():print('Cached',date,flush=True);continue
        benchmark,_,_=samples(date);source,rgb,ids=dense_training(date,benchmark)
        np.testing.assert_array_equal(ids,np.load(ROOT/f'results/object_alignment/{date}_reference_roof/training_ids.npz')['original_ply_indices'])
        assert np.all(np.floor(source/.4).sum(1).astype(int)%4!=0)
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
        colors=np.c_[las.red,las.green,las.blue].astype(float);colors/=65535 if colors.max()>255 else 255
        records={}
        for side,points,color in [('source',source,rgb),('target',target,colors)]:
            candidate_path=ROOT/f'results/object_consensus/{date}/{side}_candidates.json'
            detection=json.loads(candidate_path.read_text());up=detection['ground_plane_normalized'][:3]
            image,index,metadata=rasterize(points,color,up,PARAMETERS['raster_size'])
            probability=segmenter.predict(image);data=visible_labels(points,index,probability)
            if side=='source':data['original_ply_indices']=ids[data['input_indices']]
            np.savez_compressed(folder/f'{side}_labels.npz',**data)
            save_images(folder,side,image,index,probability)
            counts=np.bincount(data['labels'],minlength=len(GROUPS))
            records[side]={**metadata,'class_counts':dict(zip(GROUPS,counts.tolist())),
                'ground_plane_source':str(candidate_path.relative_to(ROOT)),'ground_plane_sha256':file_hash(candidate_path),
                'data_sha256':file_hash(folder/f'{side}_labels.npz')}
            print(date,side,records[side]['class_counts'],flush=True)
        record={'date':date,'seconds':time.monotonic()-begin,'clouds':records,'training_tile_leakage_count':0,
            'inputs':{f'assets/{date}/{date}.{ext}':file_hash(ROOT/f'assets/{date}/{date}.{ext}') for ext in ['las','ply']}}
        (folder/'segmentation.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Finished segmentation',date,round(record['seconds'],1),'s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='*',default=DATES);main(p.parse_args().dates)
