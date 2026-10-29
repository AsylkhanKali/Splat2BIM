"""Recompute trial metrics and verify that prior evidence remains unchanged."""
import csv
import json

import laspy
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES,samples
from align_0929 import apply,summary
from align_object_recovery import OUT,load_fragments,surviving_pairs,eligible_candidates,scale_admissible,SCALE_BAND
from align_object_consensus import patch_checks
from object_consensus import descriptors,spatial_support
from check_static_structures import volume_mask
from check_las_vertical import file_hash
from select_object_alignment import choose


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['scale_band']==list(SCALE_BAND)
    for name,sha in protocol['code_sha256'].items():assert file_hash(ROOT/'scripts'/name)==sha
    manifest=json.loads((OUT/'inputs.json').read_text());files={}
    for value in manifest.values():
        for f in value['files']:files[f['path']]=f['sha256']
    with (OUT/'static_structures.csv').open() as f:static_rows=list(csv.DictReader(f))
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions'];checks=[]
    for date in DATES:
        folder=OUT/date;a=json.loads((folder/'audit.json').read_text())
        assert a['code_sha256']==protocol['code_sha256'] and a['scale_band']==list(SCALE_BAND)
        src,dst=load_fragments(date);sd,td=descriptors(src),descriptors(dst)
        ids=np.load(ROOT/a['training_ids_path'])['original_ply_indices']
        v=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data[ids]
        xyz=np.c_[v['x'],v['y'],v['z']].astype(float)
        assert np.all(ids%2==0) and np.all(np.floor(xyz/.4).sum(1).astype(int)%4!=0)
        benchmark,_,held=samples(date);data=np.load(folder/'held_out_distances.npz')
        previous=np.load(ROOT/f'results/object_consensus/{date}/held_out_distances.npz')
        np.testing.assert_array_equal(data['source_points'],benchmark[held])
        np.testing.assert_array_equal(data['benchmark_sample_indices'],np.flatnonzero(held))
        np.testing.assert_array_equal(data['fixed_coverage'],previous['fixed_coverage'])
        cache=a['initial_hypotheses'].get('cache_path');cached=None
        if cache:
            assert file_hash(ROOT/cache)==a['initial_hypotheses']['cache_sha256']
            cached=np.load(ROOT/cache)['matrices']
        for c in a['coarse_candidates']:
            assert scale_admissible(c['initial_matrix'],a['extent_scale_prior'])
            if cached is not None:np.testing.assert_array_equal(c['initial_matrix'],cached[c['cached_seed_index']])
            pairs=surviving_pairs(c['initial_pairs'],sd,td,np.array(c['matrix']))
            support=spatial_support(pairs,sd,td,np.array(c['matrix']))
            admissible=scale_admissible(c['matrix'],a['extent_scale_prior'])
            assert c['scale_admissible']==admissible
            assert c['coarse_support_count']==(len(support) if admissible else 0)
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN;tree=cKDTree(target)
        matrices={}
        for r in a['runs']:
            eligible=list(range(len(a['coarse_candidates']))) if r['method']=='geometric_control' else eligible_candidates(a['coarse_candidates'])
            if not eligible:
                assert 'matrix_local' not in r;continue
            expected_index=min(eligible,key=lambda i:a['coarse_candidates'][i]['training_coarse_median_m'])
            assert r['selected_seed_index']==expected_index
            chosen=a['coarse_candidates'][expected_index];m=np.array(r['matrix_local']);matrices[r['method']]=m
            pairs=patch_checks(chosen['initial_pairs'],src,dst,m)
            assert len(pairs)==len(r['final_pair_checks'])
            for p,saved in zip(pairs,r['final_pair_checks']):
                assert (p['source_id'],p['target_id'])==(saved['source_id'],saved['target_id'])
                for key in ['median_m','p90_m','fraction_within_1m','reverse_fraction_within_1m']:np.testing.assert_allclose(p[key],saved[key],atol=1e-9)
            current={(p['source_id'],p['target_id']):p for p in surviving_pairs(chosen['initial_pairs'],sd,td,m)}
            valid=[{**p,'center_distance_m':current[(p['source_id'],p['target_id'])]['center_distance_m']} for p in pairs
                if (p['source_id'],p['target_id']) in current and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
            support=spatial_support(valid,sd,td,m)
            assert [(p['source_id'],p['target_id']) for p in support]==[(p['source_id'],p['target_id']) for p in r['final_spatial_support']]
            assert len(spatial_support([p for p in valid if p['fraction_within_1m']>=.8],sd,td,m))==r['strict_forward_080_spatial_support_count']
            if r['status']=='candidate_for_review':assert support and scale_admissible(m,a['extent_scale_prior'])
        old=json.loads((ROOT/f'results/object_consensus/{date}/audit.json').read_text())
        selected=json.loads((folder/'selection.json').read_text())
        chosen_method,chosen_run=choose(old['runs'][1],a['runs'][1])
        assert selected['selected_method']==chosen_method
        assert selected['selector_sha256']==file_hash(ROOT/'scripts/select_object_alignment.py')
        for f in selected['input_audits']:assert file_hash(ROOT/f['path'])==f['sha256']
        if chosen_run:
            np.testing.assert_array_equal(selected['matrix_local'],chosen_run['matrix_local'])
            matrices['hybrid']=np.array(selected['matrix_local'])
        matrices['previous_consensus']=np.array(old['runs'][1]['matrix_local'])
        base=ROOT/('results/0929_baseline' if date=='0929' else f'results/multidate/{date}')
        matrices['working_baseline']=np.array(json.loads((base/'alignment.json').read_text())['ply_to_las_local'])
        controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz')
        for method,m in matrices.items():
            scale=np.cbrt(np.linalg.det(m[:3,:3]));assert scale>0
            np.testing.assert_allclose(m[:3,:3].T@m[:3,:3],np.eye(3)*scale**2,atol=1e-8)
            np.testing.assert_array_equal(m[3],[0,0,0,1])
            errors=tree.query(apply(benchmark[held],m))[0]
            saved_method=chosen_method if method=='hybrid' else method
            np.testing.assert_allclose(errors,data[saved_method],atol=1e-9)
            for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',data['fixed_coverage'])]:
                saved=next(row for row in a['comparison'] if row['method']==saved_method and row['scope']==scope)
                for key,value in summary(errors[mask]).items():np.testing.assert_allclose(value,saved[key],atol=1e-9)
            for region in regions:
                local=cKDTree(target[volume_mask(target,region)])
                for scope in ['unseen_records','spatial_holdout']:
                    p=controls[region['id']+'_'+scope]
                    row=next(r for r in static_rows if r['date']==date and r['method']==method and r['region']==region['id'] and r['scope']==scope)
                    assert int(row['count'])==len(p)
                    if len(p):
                        for key,value in summary(local.query(apply(p,m))[0]).items():np.testing.assert_allclose(value,float(row[key]),atol=1e-9)
        checks.append({'date':date,'matrices_recomputed':len(matrices),'training_records_checked':len(ids),
            'withheld_records_checked':int(held.sum())});print('Verified',date,flush=True)
    for path,sha in files.items():assert file_hash(ROOT/path)==sha,path
    tests=(OUT/'tests.log').read_text();assert 'Ran 39 tests' in tests and tests.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':39,'unchanged_file_hashes':len(files),
        'original_input_hashes_matched':sum(p.startswith('assets/') for p in files),
        'working_registration_files_unchanged':sum(p.endswith(('alignment.json','T_3dgs_to_las_local.txt','T_3dgs_to_las_global.txt')) for p in files),
        'frozen_code_and_scale_band_match':True,'training_tile_leakage_count':0,'date_checks':checks,
        'meaning':'Numerical and provenance checks, not independent survey validation.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('All checks passed.')


if __name__=='__main__':main()
