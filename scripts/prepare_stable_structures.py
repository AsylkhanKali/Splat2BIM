"""Freeze measured surface variants and independent LAS temporal persistence."""
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN,samples
from align_objects import dense_training
from align_object_recovery import load_fragments
from structural_surfaces import PARAMETERS,serializable
from structural_correspondence import combine_surfaces
from surface_infill import infill_surfaces,add_infill
from prepare_structural_surfaces import load_surfaces,OUT as GROWN
from prepare_temporal_surfaces import TEMPORAL_PARAMETERS
from check_las_vertical import file_hash

OUT=ROOT/'results/stable_structures'
SOURCES=['prepare_stable_structures.py','surface_infill.py','structural_surfaces.py',
    'structural_correspondence.py','object_candidates.py','object_consensus.py']


def load_structures(date,side):
    folder=OUT/date;found=json.loads((folder/f'{side}_structures.json').read_text())
    points=np.load(folder/f'{side}_structure_points.npz')
    for row in found['candidates']:row['points']=(points[row['id']]-found['normalization_center'])*found['normalization_scale']
    return found


def build(date):
    benchmark,_,_=samples(date);source,_,ids=dense_training(date,benchmark)
    assert np.all(ids%2==0) and np.all(np.floor(source/.4).sum(1).astype(int)%4!=0)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    originals=load_fragments(date);combined={};inputs={}
    for side,points,old,labels_folder in [('source',source,originals[0],'semantic_3dgs'),('target',target,originals[1],'semantic_fusion')]:
        label_path=ROOT/f'results/{labels_folder}/{date}/{side}_labels.npz';labels=dict(np.load(label_path))
        combined[side]=add_infill(combine_surfaces(old,load_surfaces(date,side)),infill_surfaces(points,labels,old))
        paths=[label_path]+[GROWN/date/f'{side}_{suffix}' for suffix in ['surfaces.json','surface_points.npz']]
        paths+=[ROOT/f'results/object_consensus/{date}/{side}_{suffix}' for suffix in ['candidates.json','candidate_points.npz']]
        inputs.update({str(p.relative_to(ROOT)):file_hash(p) for p in paths})
    for ext in ['ply','las']:
        path=ROOT/f'assets/{date}/{date}.{ext}';inputs[str(path.relative_to(ROOT))]=file_hash(path)
    return combined,inputs


def main():
    OUT.mkdir(exist_ok=True)
    protocol={'parameters':PARAMETERS,'temporal_parameters':TEMPORAL_PARAMETERS,
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in SOURCES},
        'source_training_only':True,'source_alignment_matrices_used':False,'control_regions_used':False,
        'variants':['original_white','connected_full_color','measured_infill_within_original_footprint'],
        'one_support_per_source_and_target_family':True,'all_seven_dates_same_rules':True,
        'growth_protocol_sha256':file_hash(GROWN/'extraction_protocol.json')}
    path=OUT/'extraction_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Preserve frozen experiment.'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    if (OUT/'extraction.json').exists():print('Cached all surfaces');return
    combined={};inputs={};coverage={}
    for date in DATES:
        combined[date],date_inputs=build(date);inputs.update(date_inputs)
        coverage[date]={r['id']:{} for r in combined[date]['target']['candidates']}
        print('Extracted',date,{s:len(f['candidates']) for s,f in combined[date].items()},flush=True)
    for reference in DATES:
        las=laspy.read(ROOT/f'assets/{reference}/{reference}.las');tree=cKDTree(np.c_[las.x,las.y,las.z]-ORIGIN)
        for date in DATES:
            found=combined[date]['target']
            for row in found['candidates']:
                points=row['points']/found['normalization_scale']+found['normalization_center']
                coverage[date][row['id']][reference]=float(np.mean(tree.query(points)[0]<TEMPORAL_PARAMETERS['distance_m']))
        print('Temporal reference',reference,flush=True)
    summary=[]
    for date in DATES:
        folder=OUT/date;folder.mkdir(exist_ok=True)
        for side in ['source','target']:
            found=combined[date][side];before=len(found['candidates'])
            if side=='target':
                for row in found['candidates']:
                    row['temporal_coverage']=coverage[date][row['id']]
                    row['supported_dates']=[d for d,v in row['temporal_coverage'].items() if v>=TEMPORAL_PARAMETERS['minimum_coverage']]
                found['candidates']=[r for r in found['candidates'] if date in r['supported_dates'] and len(r['supported_dates'])>=TEMPORAL_PARAMETERS['minimum_dates_including_capture']]
            (folder/f'{side}_structures.json').write_text(json.dumps(serializable(found),indent=2)+'\n')
            np.savez_compressed(folder/f'{side}_structure_points.npz',**{r['id']:r['points']/found['normalization_scale']+found['normalization_center'] for r in found['candidates']})
            summary.append({'date':date,'side':side,'alternatives_before_persistence':before,'alternatives':len(found['candidates']),
                'families':len({r['family'] for r in found['candidates']}),
                'metadata_sha256':file_hash(folder/f'{side}_structures.json'),'points_sha256':file_hash(folder/f'{side}_structure_points.npz')})
            print(date,side,before,'→',len(found['candidates']),summary[-1]['families'],'families',flush=True)
    (OUT/'extraction.json').write_text(json.dumps({'inputs':inputs,'clouds':summary,'target_coverage_all_alternatives':coverage},indent=2)+'\n')


if __name__=='__main__':main()
