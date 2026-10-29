"""Require each target surface alternative to recur in at least three LAS dates."""
import json
import itertools

import numpy as np

from align_dates import ROOT,DATES
from align_object_recovery import load_fragments
from align_object_consensus import patch_checks
from object_consensus import descriptors,PARAMETERS as GATES
from structural_surfaces import PARAMETERS,serializable
from structural_correspondence import combine_surfaces,gated_pairs
from prepare_structural_surfaces import OUT,load_surfaces
from check_las_vertical import file_hash


def load_families(date,side):
    folder=OUT/date;found=json.loads((folder/f'{side}_families.json').read_text());points=np.load(folder/f'{side}_family_points.npz')
    for row in found['candidates']:row['points']=(points[row['id']]-found['normalization_center'])*found['normalization_scale']
    return found


def main():
    combined={};evidence={}
    for date in DATES:
        s,t=load_fragments(date)
        combined[date]={side:combine_surfaces(old,load_surfaces(date,side)) for side,old in [('source',s),('target',t)]}
        evidence[date]={r['id']:[] for r in combined[date]['target']['candidates']}
    for a,b in itertools.combinations(DATES,2):
        s,t=combined[a]['target'],combined[b]['target']
        pairs=patch_checks(gated_pairs(descriptors(s),descriptors(t),np.eye(4)),s,t,np.eye(4))
        for pair in pairs:
            if min(pair['fraction_within_1m'],pair['reverse_fraction_within_1m'])<.5:continue
            evidence[a][pair['source_id']].append({'date':b,'id':pair['target_id'],'forward':pair['fraction_within_1m'],'reverse':pair['reverse_fraction_within_1m'],'center_distance_m':pair['center_distance_m']})
            evidence[b][pair['target_id']].append({'date':a,'id':pair['source_id'],'forward':pair['reverse_fraction_within_1m'],'reverse':pair['fraction_within_1m'],'center_distance_m':pair['center_distance_m']})
    inputs={};summary=[]
    for date in DATES:
        for side in ['source','target']:
            found=combined[date][side];before=len(found['candidates'])
            if side=='target':
                for row in found['candidates']:
                    row['persistence_evidence']=evidence[date][row['id']]
                    row['supported_dates']=sorted({date}|{r['date'] for r in row['persistence_evidence']})
                found['candidates']=[r for r in found['candidates'] if len(r['supported_dates'])>=PARAMETERS['minimum_persistent_dates']]
            folder=OUT/date
            (folder/f'{side}_families.json').write_text(json.dumps(serializable(found),indent=2)+'\n')
            np.savez_compressed(folder/f'{side}_family_points.npz',**{r['id']:r['points']/found['normalization_scale']+found['normalization_center'] for r in found['candidates']})
            for suffix in ['surfaces.json','surface_points.npz']:
                p=folder/f'{side}_{suffix}';inputs[str(p.relative_to(ROOT))]=file_hash(p)
            summary.append({'date':date,'side':side,'alternatives_before_persistence':before,'alternatives':len(found['candidates']),
                'families':len({r['family'] for r in found['candidates']}),
                'metadata_sha256':file_hash(folder/f'{side}_families.json'),'points_sha256':file_hash(folder/f'{side}_family_points.npz')})
            print(summary[-1],flush=True)
    (OUT/'persistence.json').write_text(json.dumps({'minimum_dates':PARAMETERS['minimum_persistent_dates'],
        'distance_m':1.,'minimum_fraction_both_directions':.5,'geometry_gates':GATES,
        'uses_provided_las_frame_only':True,'source_alignment_matrices_used':False,
        'evidence_for_all_target_alternatives':evidence,'inputs':inputs,'clouds':summary,
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['prepare_persistent_surfaces.py','structural_correspondence.py','structural_surfaces.py','object_consensus.py','object_candidates.py']}},indent=2)+'\n')


if __name__=='__main__':main()
