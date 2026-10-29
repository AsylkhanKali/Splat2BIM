"""Supervised local refinement using existing static-region definitions and train records only."""
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN,samples
from align_0929 import apply,summary
from align_objects import dense_training
from align_stable_structures import support_evidence
from prepare_stable_structures import load_structures,OUT as BASE
from structural_pose_refinement import PARAMETERS,sample_indices,target_surface,metrics,refine,select_candidate,extent_anchors
from guarded_structural_pose import fractional_correction
from check_static_structures import volume_mask
from check_las_vertical import file_hash

OUT=ROOT/'results/reviewed_structural_refinement'
FRACTIONS=[1.,.5,.25,.125,.0625,.03125]


def prepare_groups(train,ids,target,initial,regions):
    moved=apply(train,initial);groups=[];metadata=[];cache={}
    for region in regions:
        key=region['id'];si=np.flatnonzero(volume_mask(moved,region));ti=np.flatnonzero(volume_mask(target,region))
        si=si[sample_indices(moved[si])];ti=ti[sample_indices(target[ti])]
        block=np.floor(moved[si]/PARAMETERS['calibration_block_m']).astype(int)
        calibration=(block@np.array([1,3,7]))%PARAMETERS['calibration_modulus']==0
        fi,ci=si[~calibration],si[calibration]
        if len(fi)<PARAMETERS['minimum_fit_points'] or len(ci)<PARAMETERS['minimum_calibration_points']:continue
        groups.append({'family':key,'fit':train[fi],'calibration':train[ci],'target':target_surface(target[ti])})
        metadata.append({'key':key,'family':key,'fit_points':len(fi),'calibration_points':len(ci),'target_points':len(ti)})
        for name,value in [('fit',train[fi]),('calibration',train[ci]),('target',target[ti]),
            ('fit_ply_ids',ids[fi]),('calibration_ply_ids',ids[ci]),('target_las_ids',ti)]:cache[key+'_'+name]=value
    cache['anchors']=extent_anchors(train)
    return groups,metadata,cache


def run(date,regions):
    folder=OUT/date;folder.mkdir(exist_ok=True);parent=json.loads((BASE/date/'audit.json').read_text());old=parent['run']
    initial=np.array(old['matrix_local']);bank=parent['coarse_candidates'][old['selected_candidate_index']]['initial_pairs']
    sf,tf=load_structures(date,'source'),load_structures(date,'target');benchmark,rgb,held=samples(date)
    train,_,ids=dense_training(date,benchmark);assert np.all(ids%2==0) and np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255;tree=cKDTree(target)
    groups,metadata,cache=prepare_groups(train,ids,target,initial,regions);np.savez_compressed(folder/'fit_data.npz',**cache)
    baseline=metrics(groups,initial);baseline_training=summary(tree.query(apply(benchmark[~held],initial))[0])
    anchor=apply(cache['anchors'],initial).mean(0);proposals=[];candidates=[]
    if len(groups)==len(regions):
        for weight in PARAMETERS['tangent_weights']:
            proposal,optimization=refine(groups,initial,weight,cache['anchors']);proposals.append({'tangent_weight':weight,'matrix':proposal.tolist(),'optimization':optimization})
            for fraction in FRACTIONS:
                matrix=fractional_correction(initial,proposal,fraction,anchor);error,near=tree.query(apply(benchmark[~held],matrix))
                training=summary(error);rgb_error=float(np.median(np.linalg.norm(rgb[~held]-color[near],axis=1)))
                evidence=support_evidence(bank,sf,tf,matrix)
                candidates.append({'tangent_weight':weight,'fraction':fraction,'matrix_local':matrix.tolist(),'calibration':metrics(groups,matrix),
                    'fit':metrics(groups,matrix,'fit'),'training':training,'training_rgb':rgb_error,
                    'strict_pass':bool(training['median']<=.5 and rgb_error<=.35 and evidence['strict_forward_080_spatial_support_count']>=3),
                    'global_training_guard':bool(training['median']<=baseline_training['median']+PARAMETERS['maximum_training_median_regression_m']),**evidence})
    chosen=select_candidate(candidates,baseline) if candidates else None;matrix=initial if chosen is None else np.array(candidates[chosen]['matrix_local'])
    paths=[BASE/date/'audit.json',BASE/'protocol.json',ROOT/'results/static_structures/regions.json']
    paths+=[ROOT/f'assets/{date}/{date}.{ext}' for ext in ['ply','las']]
    audit={'date':date,'initial_matrix':initial.tolist(),'matrix_local':matrix.tolist(),'selected_candidate':chosen,
        'status':'refined' if chosen is not None else 'baseline_retained','groups':metadata,'baseline_calibration':baseline,
        'baseline_training':baseline_training,'proposals':proposals,'candidates':candidates,'fixed_prefit_bank':bank,
        'held_or_manual_control_errors_used_for_selection':False,'manual_region_definitions_used_for_training':True,'scale_fixed_to_parent':True,
        'fitting_data_sha256':file_hash(folder/'fit_data.npz'),'inputs':{str(p.relative_to(ROOT)):file_hash(p) for p in paths}}
    (folder/'fit.json').write_text(json.dumps(audit,indent=2)+'\n');np.savetxt(folder/'T_3dgs_to_las_local.txt',matrix,fmt='%.17g')
    global_matrix=matrix.copy();global_matrix[:3,3]+=ORIGIN;np.savetxt(folder/'T_3dgs_to_las_global.txt',global_matrix,fmt='%.17g')
    print(date,audit['status'],'groups',[(r['fit_points'],r['calibration_points']) for r in metadata],
        'choice',None if chosen is None else (candidates[chosen]['tangent_weight'],candidates[chosen]['fraction']),flush=True)


def main():
    OUT.mkdir(exist_ok=True);parent=json.loads((BASE/'protocol.json').read_text());regions_path=ROOT/'results/static_structures/regions.json'
    for n,h in parent['source_sha256'].items():assert file_hash(ROOT/'scripts'/n)==h
    protocol={'refinement_parameters':PARAMETERS,'fractions':FRACTIONS,'object_parameters':parent['object_parameters'],'strict_forward_fraction':.8,
        'training_median_limit_m':.5,'training_rgb_limit':.35,'regions_sha256':file_hash(regions_path),'parent_protocol_sha256':file_hash(BASE/'protocol.json'),
        'same_rules_all_dates':True,'manual_region_definitions_used_for_training':True,'control_records_or_errors_used_for_selection':False,
        'rigid_correction_only':True,'same_site_development_experiment':True,'design_revised_after_automatic_refinement_evaluation':True,
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['fit_reviewed_structural_refinement.py','structural_pose_refinement.py','guarded_structural_pose.py']}}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    regions=json.loads(regions_path.read_text())['regions']
    for date in DATES:
        if (OUT/date/'fit.json').exists():print('Cached',date,flush=True);continue
        run(date,regions)


if __name__=='__main__':main()
