"""Cached fragments, pair-distance scale seeds and correspondence-aware selection."""
import argparse
import json
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,samples,refine
from align_0929 import apply,downsample,icp,summary
from align_objects import dense_training
from align_object_consensus import rank_seeds,patch_checks
from object_consensus import descriptors,match,spatial_support
from pair_distance_seeds import hypotheses,PARAMETERS
from check_las_vertical import file_hash,write_csv

OUT=ROOT/'results/object_recovery'
SCALE_BAND=(.4,1.6)


def scale_admissible(matrix,extent_scale):
    ratio=np.cbrt(np.linalg.det(np.asarray(matrix)[:3,:3]))/extent_scale
    return bool(SCALE_BAND[0]<=ratio<=SCALE_BAND[1])


def load_fragments(date):
    found={};folder=ROOT/f'results/object_consensus/{date}'
    for side in ['source','target']:
        item=json.loads((folder/f'{side}_candidates.json').read_text())
        points=np.load(folder/f'{side}_candidate_points.npz')
        for row in item['candidates']:
            row['points']=(points[row['id']]-item['normalization_center'])*item['normalization_scale']
        found[side]=item
    return found['source'],found['target']


def surviving_pairs(original,source,target,matrix):
    current={(p['source_id'],p['target_id']):p for p in match(source,target,matrix)}
    # Return current distances but never invent an identity absent from the seed.
    return [current[(p['source_id'],p['target_id'])] for p in original if (p['source_id'],p['target_id']) in current]


def eligible_candidates(candidates):
    return [i for i,c in enumerate(candidates) if c['coarse_support_count']>=3]


def run(date):
    started=time.monotonic();out=OUT/date;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/'inputs.json').read_text())[date]
    for r in manifest['files']:assert file_hash(ROOT/r['path'])==r['sha256']
    source_found,target_found=load_fragments(date);sd=descriptors(source_found);td=descriptors(target_found)
    extent_scale=source_found['normalization_scale']/target_found['normalization_scale']
    benchmark,rgb,held=samples(date);train,colors,ids=dense_training(date,benchmark)
    previous_ids=ROOT/f'results/object_alignment/{date}_reference_roof/training_ids.npz'
    np.testing.assert_array_equal(ids,np.load(previous_ids)['original_ply_indices'])
    assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    target_rgb=np.c_[las.red,las.green,las.blue].astype(float);target_rgb/=65535 if target_rgb.max()>255 else 255
    cache=ROOT/f'results/object_recovery_unbounded/{date}/initial_matrices.npz'
    if cache.exists():
        cached=np.load(cache)['matrices'];proposals=[]
        for index,matrix in enumerate(cached):
            if not scale_admissible(matrix,extent_scale):continue
            pairs=match(sd,td,matrix)
            if not spatial_support(pairs,sd,td,matrix):continue
            proposals.append({'matrix':matrix,'pairs':pairs,'anchor_source_ids':[],'anchor_target_ids':[],
                'cached_seed_index':index})
        counts={'cached_proposals':len(cached),'retained_proposals':len(proposals),
            'cache_path':str(cache.relative_to(ROOT)),'cache_sha256':file_hash(cache)}
    else:
        proposals,counts=hypotheses(source_found,target_found)
        counts['before_scale_band']=len(proposals)
        proposals=[p for p in proposals if scale_admissible(p['matrix'],extent_scale)]
        counts['retained_proposals']=len(proposals)
    np.savez_compressed(out/'initial_matrices.npz',matrices=np.array([r['matrix'] for r in proposals]))
    print(date,counts,'seed generation seconds',round(time.monotonic()-started,1),flush=True)
    small=downsample(train,.06);target_small=downsample(target,2.);tree=cKDTree(target_small)
    score=downsample(train,.12);rng=np.random.default_rng(29)
    if len(score)>2000:score=score[rng.choice(len(score),2000,replace=False)]
    chosen=rank_seeds(proposals,score,tree);candidates=[]
    for seed in chosen:
        matrix=icp(small,target_small,seed['matrix'],[40.,20.,10.,5.],with_scale=True,iterations=40)
        kept=surviving_pairs(seed['pairs'],sd,td,matrix);support=spatial_support(kept,sd,td,matrix)
        admissible=scale_admissible(matrix,extent_scale)
        candidates.append({'initial_matrix':seed['matrix'].tolist(),'matrix':matrix.tolist(),
            'initial_score_p75_m':seed['score_p75_m'],'initial_pairs':seed['pairs'],
            'anchor_source_ids':seed['anchor_source_ids'],'anchor_target_ids':seed['anchor_target_ids'],
            'cached_seed_index':seed.get('cached_seed_index'),'scale_admissible':admissible,
            'training_coarse_median_m':float(np.median(tree.query(apply(small,matrix))[0])),
            'coarse_support_count':len(support) if admissible else 0,'coarse_support':support})
    fine_source=downsample(train,.025);fine_target=downsample(target,.6);full_tree=cKDTree(target)
    runs=[];matrices={};refined={}
    for method,eligible in [('geometric_control',list(range(len(candidates)))),('pair_preserving',eligible_candidates(candidates))]:
        if not eligible:
            runs.append({'method':method,'status':'rejected_no_preserved_coarse_support'});continue
        index=min(eligible,key=lambda i:candidates[i]['training_coarse_median_m']);selected=candidates[index]
        if index not in refined:refined[index]=refine(fine_source,fine_target,np.array(selected['matrix']))
        matrix=refined[index];matrices[method]=matrix
        error,near=full_tree.query(apply(benchmark[~held],matrix));color_error=float(np.median(np.linalg.norm(rgb[~held]-target_rgb[near],axis=1)))
        checks=patch_checks(selected['initial_pairs'],source_found,target_found,matrix)
        current={(p['source_id'],p['target_id']):p for p in surviving_pairs(selected['initial_pairs'],sd,td,matrix)}
        valid=[{**p,'center_distance_m':current[(p['source_id'],p['target_id'])]['center_distance_m']} for p in checks
            if (p['source_id'],p['target_id']) in current and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
        support=spatial_support(valid,sd,td,matrix)
        strict=spatial_support([p for p in valid if p['fraction_within_1m']>=.8],sd,td,matrix)
        status='rejected_training_alignment' if np.median(error)>.5 or color_error>.35 or not scale_admissible(matrix,extent_scale) else (
            'candidate_for_review' if support else 'rejected_object_correspondences')
        runs.append({'method':method,'status':status,'selected_seed_index':index,'matrix_local':matrix.tolist(),
            'training_final':summary(error),'training_color_difference':color_error,'final_pair_checks':checks,
            'final_spatial_support':support,'strict_forward_080_spatial_support_count':len(strict)})
        print(date,method,status,'training median',round(float(np.median(error)),4),'pairs',len(support),flush=True)
    # Previous transforms are used only now, for diagnostic comparison.
    old=json.loads((ROOT/f'results/object_consensus/{date}/audit.json').read_text())
    matrices['previous_consensus']=np.array(next(r for r in old['runs'] if r['method']=='multi_fragment')['matrix_local'])
    base=ROOT/('results/0929_baseline' if date=='0929' else f'results/multidate/{date}')
    matrices['working_baseline']=np.array(json.loads((base/'alignment.json').read_text())['ply_to_las_local'])
    previous_check=np.load(ROOT/f'results/object_consensus/{date}/held_out_distances.npz')
    np.testing.assert_array_equal(benchmark[held],previous_check['source_points']);fixed=previous_check['fixed_coverage']
    rows=[];distances={}
    for method,matrix in matrices.items():
        error=full_tree.query(apply(benchmark[held],matrix))[0];distances[method]=error
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed)]:
            rows.append({'date':date,'method':method,'scope':scope,**summary(error[mask]),'within_1m':float(np.mean(error[mask]<1.))})
    np.savez_compressed(out/'held_out_distances.npz',source_points=benchmark[held],benchmark_sample_indices=np.flatnonzero(held),fixed_coverage=fixed,**distances)
    write_csv(out/'comparison.csv',rows)
    audit={'date':date,'parameters':PARAMETERS,'scale_band':SCALE_BAND,'extent_scale_prior':extent_scale,
        'cached_fragments':f'results/object_consensus/{date}',
        'initial_hypotheses':counts,'coarse_candidates':candidates,'runs':runs,'comparison':rows,
        'training_ids_path':str(previous_ids.relative_to(ROOT)),'training_tile_leakage_count':0,
        'old_matrices_used_to_generate_or_select_new_fit':False,'total_seconds':time.monotonic()-started,
        'code_sha256':{name:file_hash(ROOT/'scripts'/name) for name in ['pair_distance_seeds.py','align_object_recovery.py']}}
    for r in manifest['files']:assert file_hash(ROOT/r['path'])==r['sha256']
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='+');args=p.parse_args()
    for date in args.dates:run(date)
