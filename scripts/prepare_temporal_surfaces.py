"""Check spatial persistence against full LAS, independently of fragment boundaries."""
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN
from align_object_recovery import load_fragments
from structural_surfaces import serializable
from structural_correspondence import combine_surfaces
from prepare_structural_surfaces import OUT,load_surfaces
from check_las_vertical import file_hash

TEMPORAL_PARAMETERS={'distance_m':1.,'minimum_coverage':.8,'minimum_dates_including_capture':3}


def load_structures(date,side):
    folder=OUT/date;found=json.loads((folder/f'{side}_structures.json').read_text());points=np.load(folder/f'{side}_structure_points.npz')
    for row in found['candidates']:row['points']=(points[row['id']]-found['normalization_center'])*found['normalization_scale']
    return found


def main():
    combined={};coverage={};inputs={}
    for date in DATES:
        s,t=load_fragments(date);combined[date]={side:combine_surfaces(old,load_surfaces(date,side)) for side,old in [('source',s),('target',t)]}
        coverage[date]={r['id']:{} for r in combined[date]['target']['candidates']}
    for reference in DATES:
        path=ROOT/f'assets/{reference}/{reference}.las';las=laspy.read(path);tree=cKDTree(np.c_[las.x,las.y,las.z]-ORIGIN)
        inputs[str(path.relative_to(ROOT))]=file_hash(path)
        for date in DATES:
            found=combined[date]['target']
            for row in found['candidates']:
                points=row['points']/found['normalization_scale']+found['normalization_center']
                error=tree.query(points)[0];coverage[date][row['id']][reference]=float(np.mean(error<TEMPORAL_PARAMETERS['distance_m']))
        print('Temporal reference',reference,flush=True)
    summary=[]
    for date in DATES:
        for side in ['source','target']:
            found=combined[date][side];before=len(found['candidates'])
            if side=='target':
                for row in found['candidates']:
                    row['temporal_coverage']=coverage[date][row['id']]
                    row['supported_dates']=[d for d,v in row['temporal_coverage'].items() if v>=TEMPORAL_PARAMETERS['minimum_coverage']]
                found['candidates']=[r for r in found['candidates'] if date in r['supported_dates'] and len(r['supported_dates'])>=TEMPORAL_PARAMETERS['minimum_dates_including_capture']]
            folder=OUT/date
            (folder/f'{side}_structures.json').write_text(json.dumps(serializable(found),indent=2)+'\n')
            np.savez_compressed(folder/f'{side}_structure_points.npz',**{r['id']:r['points']/found['normalization_scale']+found['normalization_center'] for r in found['candidates']})
            for suffix in ['surfaces.json','surface_points.npz']:
                path=folder/f'{side}_{suffix}';inputs[str(path.relative_to(ROOT))]=file_hash(path)
            summary.append({'date':date,'side':side,'alternatives_before_persistence':before,'alternatives':len(found['candidates']),
                'families':len({r['family'] for r in found['candidates']}),
                'metadata_sha256':file_hash(folder/f'{side}_structures.json'),'points_sha256':file_hash(folder/f'{side}_structure_points.npz')})
            print(date,side,before,'→',len(found['candidates']),summary[-1]['families'],'families',flush=True)
    (OUT/'temporal_protocol.json').write_text(json.dumps({'parameters':TEMPORAL_PARAMETERS,'target_coverage_all_alternatives':coverage,
        'source_alignment_matrices_used':False,'uses_provided_las_frame_only':True,
        'purpose':'Additional persistence filter; full-LAS temporal coverage does not replace bidirectional within-date fragment validation.',
        'inputs':inputs,'clouds':summary,'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['prepare_temporal_surfaces.py','structural_correspondence.py','structural_surfaces.py','object_consensus.py','object_candidates.py']}},indent=2)+'\n')


if __name__=='__main__':main()
