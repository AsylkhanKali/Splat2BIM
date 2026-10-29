"""Controlled single-fragment vs multi-fragment initialization, training tiles only."""
import argparse
import json
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,samples,refine
from align_0929 import apply,downsample,icp,summary
from align_objects import dense_training
from check_las_vertical import file_hash,write_csv
from object_candidates import PARAMETERS as DETECTOR_PARAMETERS,serializable
from object_consensus import PARAMETERS,fragments,descriptors,hypotheses,match,spatial_support

OUT=ROOT/'results/object_consensus'


def patch_checks(pairs,source_found,target_found,matrix):
    source={r['id']:r['points']/source_found['normalization_scale']+source_found['normalization_center'] for r in source_found['candidates']}
    target={r['id']:r['points']/target_found['normalization_scale']+target_found['normalization_center'] for r in target_found['candidates']}
    checks=[]
    for pair in pairs:
        moved=apply(source[pair['source_id']],matrix);destination=target[pair['target_id']]
        distance=cKDTree(destination).query(moved)[0];reverse=cKDTree(moved).query(destination)[0]
        checks.append({**pair,'median_m':float(np.median(distance)),'p90_m':float(np.quantile(distance,.9)),
            'fraction_within_1m':float(np.mean(distance<1.)),
            'reverse_fraction_within_1m':float(np.mean(reverse<1.))})
    return checks


def rank_seeds(rows,score_source,target_tree):
    for row in rows:
        row['score_p75_m']=float(np.quantile(target_tree.query(apply(score_source,row['matrix']))[0],.75))
    ranked=sorted(rows,key=lambda r:r['score_p75_m']);chosen=[]
    # Four geometrically different starts, rather than four duplicate pair labels.
    anchors=score_source[::max(1,len(score_source)//20)]
    for row in ranked:
        if any(np.sqrt(np.mean(np.sum((apply(anchors,row['matrix'])-apply(anchors,c['matrix']))**2,axis=1)))<2 for c in chosen):continue
        chosen.append(row)
        if len(chosen)==4:break
    return chosen


def run(date):
    out=OUT/date;out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    base=ROOT/('results/0929_baseline' if date=='0929' else f'results/multidate/{date}')
    preserved=[{'path':str((base/name).relative_to(ROOT)),'sha256':file_hash(base/name)} for name in
        ['alignment.json','T_3dgs_to_las_local.txt','T_3dgs_to_las_global.txt']]
    inputs=[{'path':f'assets/{date}/{date}.{ext}','sha256':file_hash(ROOT/f'assets/{date}/{date}.{ext}')} for ext in ['ply','las']]
    benchmark,rgb,held=samples(date);train,colors,ids=dense_training(date,benchmark)
    old_ids=ROOT/f'results/object_alignment/{date}_reference_roof/training_ids.npz'
    np.testing.assert_array_equal(ids,np.load(old_ids)['original_ply_indices'])
    assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    target_rgb=np.c_[las.red,las.green,las.blue].astype(float);target_rgb/=65535 if target_rgb.max()>255 else 255
    source_found=fragments(train,colors);target_found=fragments(target,target_rgb)
    for side,found in [('source',source_found),('target',target_found)]:
        (out/f'{side}_candidates.json').write_text(json.dumps(serializable(found),indent=2)+'\n')
        np.savez_compressed(out/f'{side}_candidate_points.npz',**{r['id']:r['points']/found['normalization_scale']+found['normalization_center'] for r in found['candidates']})
    source_desc=descriptors(source_found);target_desc=descriptors(target_found)
    singles,multi=hypotheses(source_found,target_found)
    print(date,'fragments',len(source_desc['ids']),len(target_desc['ids']),'single seeds',len(singles),'multi seeds',len(multi),flush=True)
    np.savez_compressed(out/'initial_matrices.npz',single=np.array([r['matrix'] for r in singles]),
        multi=np.array([r['matrix'] for r in multi]),multi_seed_indices=np.array([r['seed_index'] for r in multi]))
    small_source=downsample(train,.06);small_target=downsample(target,2.);coarse_tree=cKDTree(small_target)
    score_source=downsample(train,.12);rng=np.random.default_rng(29)
    if len(score_source)>2000:score_source=score_source[rng.choice(len(score_source),2000,replace=False)]
    fine_source=downsample(train,.025);fine_target=downsample(target,.6);full_tree=cKDTree(target)
    runs=[];matrices={}
    for method,rows in [('single_fragment',singles),('multi_fragment',multi)]:
        begin=time.monotonic()
        if not rows:
            runs.append({'method':method,'status':'rejected_no_spatial_consensus','initial_hypotheses':0});continue
        chosen=rank_seeds(rows,score_source,coarse_tree);candidates=[]
        for seed in chosen:
            fitted=icp(small_source,small_target,seed['matrix'],[40.,20.,10.,5.],with_scale=True,iterations=40)
            candidates.append({'matrix':fitted,'training_coarse_median_m':float(np.median(coarse_tree.query(apply(small_source,fitted))[0])),
                'initial_matrix':seed['matrix'],'initial_score_p75_m':seed['score_p75_m'],
                'initial_pairs':seed.get('pairs',[{'source_id':seed.get('source_id'),'target_id':seed.get('target_id'),
                    'source_index':source_desc['ids'].index(seed['source_id']),'target_index':target_desc['ids'].index(seed['target_id']),
                    'center_distance_m':0.}] if 'source_id' in seed else [])})
        selected=min(candidates,key=lambda r:r['training_coarse_median_m'])
        matrix=refine(fine_source,fine_target,selected['matrix']);matrices[method]=matrix
        error,near=full_tree.query(apply(benchmark[~held],matrix))
        color_error=float(np.median(np.linalg.norm(rgb[~held]-target_rgb[near],axis=1)))
        # Preserve proposed identities: post-ICP nearest fragments cannot replace them.
        checks=patch_checks(selected['initial_pairs'],source_found,target_found,matrix)
        geometric_pairs={(r['source_id'],r['target_id']) for r in match(source_desc,target_desc,matrix)}
        valid=[r for r in checks if min(r['fraction_within_1m'],r['reverse_fraction_within_1m'])>=PARAMETERS['min_patch_fraction_each_direction'] and
            (r['source_id'],r['target_id']) in geometric_pairs]
        support=spatial_support(valid,source_desc,target_desc,matrix)
        strict_support=spatial_support([r for r in valid if r['fraction_within_1m']>=.8],source_desc,target_desc,matrix)
        training_pass=np.median(error)<=.5 and color_error<=.35
        status='rejected_training_alignment' if not training_pass else (
            ('candidate_for_review' if support else 'rejected_object_correspondences') if method=='multi_fragment' else
            ('single_pair_consistent' if valid else 'rejected_object_correspondences'))
        run={'method':method,'status':status,'initial_hypotheses':len(rows),'training_final':summary(error),
            'training_color_difference':color_error,'matrix_local':matrix.tolist(),'selected_initial_matrix':selected['initial_matrix'].tolist(),
            'selected_initial_pairs':selected['initial_pairs'],'final_pair_checks':checks,'final_spatial_support':support,
            'strict_forward_080_spatial_support_count':len(strict_support),
            'coarse_candidates':[{**c,'matrix':c['matrix'].tolist(),'initial_matrix':c['initial_matrix'].tolist()} for c in candidates],
            'seconds':time.monotonic()-begin}
        runs.append(run);print(date,method,status,'training median',np.median(error),'supported pairs',len(support),flush=True)
    # Read old transform only once new results and all selection decisions are fixed.
    baseline=np.array(json.loads((base/'alignment.json').read_text())['ply_to_las_local']);matrices['existing_baseline']=baseline
    fixed=cKDTree(target[:,:2]).query(apply(benchmark[held],baseline)[:,:2])[0]<1.
    comparison=[];distances={}
    for method,matrix in matrices.items():
        error=full_tree.query(apply(benchmark[held],matrix))[0];distances[method]=error
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed)]:
            comparison.append({'date':date,'method':method,'scope':scope,**summary(error[mask]),'within_1m':float(np.mean(error[mask]<1.))})
    np.savez_compressed(out/'held_out_distances.npz',source_points=benchmark[held],
        benchmark_sample_indices=np.flatnonzero(held),fixed_coverage=fixed,**distances)
    write_csv(out/'comparison.csv',comparison)
    audit={'date':date,'parameters':PARAMETERS,'detector_parameters':DETECTOR_PARAMETERS,
        'original_inputs':inputs,'preserved_registration_files':preserved,'source_fragment_count':len(source_desc['ids']),
        'target_fragment_count':len(target_desc['ids']),'training_ids':str(old_ids.relative_to(ROOT)),
        'training_ids_sha256':file_hash(old_ids),'training_tile_leakage_count':0,'training_records':len(train),
        'baseline_used_to_generate_transforms':False,'manual_reference_annotation_used':False,
        'original_working_clouds_changed':False,'runs':runs,'comparison':comparison,'total_seconds':time.monotonic()-started,
        'code_sha256':{name:file_hash(ROOT/'scripts'/name) for name in ['object_consensus.py','align_object_consensus.py']}}
    for item in preserved+inputs:assert file_hash(ROOT/item['path'])==item['sha256']
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dates',nargs='+');args=parser.parse_args()
    for date in args.dates:run(date)
