"""Cache independent surface growth; no registration matrices or control regions."""
import argparse
import json
import time

import laspy
import numpy as np

from align_dates import ROOT,ORIGIN,DATES,samples
from align_objects import dense_training
from align_object_recovery import load_fragments
from check_las_vertical import file_hash
from structural_surfaces import PARAMETERS,grow_surfaces,serializable

OUT=ROOT/'results/structural_surfaces'


def load_surfaces(date,side):
    folder=OUT/date;found=json.loads((folder/f'{side}_surfaces.json').read_text());points=np.load(folder/f'{side}_surface_points.npz')
    for row in found['candidates']:row['points']=(points[row['id']]-found['normalization_center'])*found['normalization_scale']
    return found


def main(dates):
    OUT.mkdir(exist_ok=True)
    protocol={'parameters':PARAMETERS,'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['structural_surfaces.py','prepare_structural_surfaces.py']},
        'alignment_matrices_used':False,'control_regions_used':False,'only_source_training_tiles':True}
    path=OUT/'extraction_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Preserve the extraction experiment.'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    for date in dates:
        start=time.monotonic();folder=OUT/date;folder.mkdir(exist_ok=True)
        if (folder/'extraction.json').exists():print('Cached',date,flush=True);continue
        benchmark,_,_=samples(date);source,_,ids=dense_training(date,benchmark)
        assert np.all(ids%2==0) and np.all(np.floor(source/.4).sum(1).astype(int)%4!=0)
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN;sf,tf=load_fragments(date)
        inputs={};records={}
        for side,points,found,labels_folder in [('source',source,sf,'semantic_3dgs'),('target',target,tf,'semantic_fusion')]:
            label_path=ROOT/f'results/{labels_folder}/{date}/{side}_labels.npz';labels=dict(np.load(label_path))
            grown=grow_surfaces(points,labels,found)
            (folder/f'{side}_surfaces.json').write_text(json.dumps(serializable(grown),indent=2)+'\n')
            np.savez_compressed(folder/f'{side}_surface_points.npz',**{r['id']:r['points']/grown['normalization_scale']+grown['normalization_center'] for r in grown['candidates']})
            inputs[str(label_path.relative_to(ROOT))]=file_hash(label_path)
            for suffix in ['candidates.json','candidate_points.npz']:
                p=ROOT/f'results/object_consensus/{date}/{side}_{suffix}';inputs[str(p.relative_to(ROOT))]=file_hash(p)
            records[side]={'seeds':len(found['candidates']),'surfaces':len(grown['candidates']),'point_counts':[len(r['points']) for r in grown['candidates']],
                'metadata_sha256':file_hash(folder/f'{side}_surfaces.json'),'points_sha256':file_hash(folder/f'{side}_surface_points.npz')}
            print(date,side,'seeds',len(found['candidates']),'surfaces',len(grown['candidates']),'seconds',round(time.monotonic()-start,1),flush=True)
        for ext in ['ply','las']:
            p=ROOT/f'assets/{date}/{date}.{ext}';inputs[str(p.relative_to(ROOT))]=file_hash(p)
        (folder/'extraction.json').write_text(json.dumps({'date':date,'inputs':inputs,'clouds':records,'seconds':time.monotonic()-start},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='*',default=DATES);main(p.parse_args().dates)
