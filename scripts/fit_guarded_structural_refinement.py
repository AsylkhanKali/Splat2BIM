"""Freeze context-anchored corrections with global training median/P90 safeguards."""
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN,samples
from align_0929 import apply,summary
from align_objects import dense_training
from align_stable_structures import support_evidence
from prepare_stable_structures import load_structures,OUT as BASE
from fit_structural_refinement import OUT as FIRST,usable_semantics
from structural_pose_refinement import PARAMETERS,target_surface,metrics,select_candidate
from guarded_structural_pose import GUARD_PARAMETERS,context_indices,refine_with_context,fractional_correction,global_guard
from check_las_vertical import file_hash

OUT=ROOT/'results/guarded_structural_refinement'


def load_groups(date):
    old=json.loads((FIRST/date/'fit.json').read_text());cache=np.load(FIRST/date/'fit_data.npz');groups=[]
    for row in old['groups']:
        key=row['key'];groups.append({'family':row['family'],'fit':cache[key+'_fit'],'calibration':cache[key+'_calibration'],
            'target':target_surface(cache[key+'_target'])})
    return old,cache,groups


def run(date):
    folder=OUT/date;folder.mkdir(exist_ok=True);first,cache,groups=load_groups(date)
    initial=np.array(first['initial_matrix']);sf,tf=load_structures(date,'source'),load_structures(date,'target')
    benchmark,rgb,held=samples(date);train,_,ids=dense_training(date,benchmark)
    labels=dict(np.load(ROOT/f'results/semantic_3dgs/{date}/source_labels.npz'))
    source_ids=np.flatnonzero(usable_semantics(train,labels,sf));source_ids=source_ids[context_indices(apply(train[source_ids],initial))]
    block=np.floor(apply(train[source_ids],initial)/PARAMETERS['calibration_block_m']).astype(int)
    source_ids=source_ids[(block@np.array([1,3,7]))%PARAMETERS['calibration_modulus']!=0]
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
    target_labels=dict(np.load(ROOT/f'results/semantic_fusion/{date}/target_labels.npz'))
    target_ids=np.flatnonzero(usable_semantics(target,target_labels,tf));context={'fit':train[source_ids],'target':{'points':target[target_ids],'tree':cKDTree(target[target_ids])}}
    np.savez_compressed(folder/'context.npz',source=train[source_ids],source_ply_ids=ids[source_ids],target_las_ids=target_ids)
    full=cKDTree(target);baseline_training=summary(full.query(apply(benchmark[~held],initial))[0]);baseline=metrics(groups,initial)
    proposal,optimization=refine_with_context(groups,context,initial,cache['anchors'])
    anchor=apply(cache['anchors'],initial).mean(0);candidates=[]
    for fraction in GUARD_PARAMETERS['fractions']:
        matrix=fractional_correction(initial,proposal,fraction,anchor)
        error,near=full.query(apply(benchmark[~held],matrix));training=summary(error)
        rgb_error=float(np.median(np.linalg.norm(rgb[~held]-color[near],axis=1)))
        evidence=support_evidence(first['fixed_prefit_bank'],sf,tf,matrix)
        strict=training['median']<=.5 and rgb_error<=.35 and evidence['strict_forward_080_spatial_support_count']>=3
        candidates.append({'fraction':fraction,'tangent_weight':GUARD_PARAMETERS['structure_tangent_weight'],'matrix_local':matrix.tolist(),
            'calibration':metrics(groups,matrix),'fit':metrics(groups,matrix,'fit'),'training':training,'training_rgb':rgb_error,
            'strict_pass':bool(strict),'global_training_guard':global_guard(baseline_training,training),**evidence})
    chosen=select_candidate(candidates,baseline);matrix=initial if chosen is None else np.array(candidates[chosen]['matrix_local'])
    audit={'date':date,'initial_matrix':initial.tolist(),'matrix_local':matrix.tolist(),'selected_candidate':chosen,
        'status':'refined' if chosen is not None else 'baseline_retained','groups':first['groups'],'baseline_calibration':baseline,
        'baseline_training':baseline_training,'candidates':candidates,'proposal_matrix':proposal.tolist(),'optimization':optimization,
        'fixed_prefit_bank':first['fixed_prefit_bank'],'held_or_manual_control_errors_used_for_selection':False,'scale_fixed_to_parent':True,
        'inputs':{str((FIRST/date/'fit.json').relative_to(ROOT)):file_hash(FIRST/date/'fit.json'),
            str((FIRST/date/'fit_data.npz').relative_to(ROOT)):file_hash(FIRST/date/'fit_data.npz')},'context_sha256':file_hash(folder/'context.npz')}
    (folder/'fit.json').write_text(json.dumps(audit,indent=2)+'\n');np.savetxt(folder/'T_3dgs_to_las_local.txt',matrix,fmt='%.17g')
    global_matrix=matrix.copy();global_matrix[:3,3]+=ORIGIN;np.savetxt(folder/'T_3dgs_to_las_global.txt',global_matrix,fmt='%.17g')
    print(date,audit['status'],'fraction',None if chosen is None else candidates[chosen]['fraction'],
        'calibration',round(baseline['loss'],6),None if chosen is None else round(candidates[chosen]['calibration']['loss'],6),flush=True)
    for c in candidates:print('  ',c['fraction'],'global guard',c['global_training_guard'],'strict',c['strict_pass'],flush=True)


def main():
    OUT.mkdir(exist_ok=True);first=json.loads((FIRST/'protocol.json').read_text())
    for name,h in first['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==h
    protocol={'refinement_parameters':PARAMETERS,'guard_parameters':GUARD_PARAMETERS,'object_parameters':first['object_parameters'],
        'strict_forward_fraction':.8,'training_median_limit_m':.5,'training_rgb_limit':.35,
        'parent_protocol_sha256':file_hash(BASE/'protocol.json'),'first_experiment_protocol_sha256':file_hash(FIRST/'protocol.json'),
        'selection_uses_held_out_errors':False,'same_rules_all_dates':True,'rigid_correction_only':True,
        'design_revised_after_first_experiment_control_evaluation':True,'same_site_development_experiment':True,
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['fit_guarded_structural_refinement.py','guarded_structural_pose.py','structural_pose_refinement.py']}}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    for date in DATES:
        if (OUT/date/'fit.json').exists():print('Cached',date,flush=True);continue
        run(date)


if __name__=='__main__':main()
