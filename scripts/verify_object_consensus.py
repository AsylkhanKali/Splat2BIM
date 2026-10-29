"""Recompute saved diagnostics; validation does not promote trial registrations."""
import csv
import json

import laspy
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES,samples
from align_0929 import apply,summary
from align_object_consensus import OUT,patch_checks
from object_consensus import PARAMETERS,descriptors,match,spatial_support
from object_candidates import PARAMETERS as DETECTOR_PARAMETERS
from check_las_vertical import file_hash
from check_static_structures import volume_mask


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['parameters']==PARAMETERS and protocol['detector_parameters']==DETECTOR_PARAMETERS
    for name,expected in protocol['code_sha256'].items():assert file_hash(ROOT/'scripts'/name)==expected
    static_protocol=json.loads((OUT/'static_protocol.json').read_text());regions_path=ROOT/static_protocol['regions_path']
    assert file_hash(regions_path)==static_protocol['regions_sha256'];regions=json.loads(regions_path.read_text())['regions']
    with (OUT/'static_structures.csv').open() as f:static_rows=list(csv.DictReader(f))
    inputs={};preserved={};checks=[]
    for date in DATES:
        folder=OUT/date;a=json.loads((folder/'audit.json').read_text())
        assert a['code_sha256']==protocol['code_sha256'] and a['parameters']==PARAMETERS
        for item in a['original_inputs']:inputs[item['path']]=item['sha256']
        for item in a['preserved_registration_files']:preserved[item['path']]=item['sha256']
        vertices=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data
        train_path=ROOT/a['training_ids'];assert file_hash(train_path)==a['training_ids_sha256']
        ids=np.load(train_path)['original_ply_indices'];v=vertices[ids];train=np.c_[v['x'],v['y'],v['z']].astype(float)
        assert len(train)==a['training_records'] and np.all(ids%2==0)
        assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
        benchmark,_,held=samples(date);data=np.load(folder/'held_out_distances.npz')
        np.testing.assert_array_equal(data['source_points'],benchmark[held])
        np.testing.assert_array_equal(data['benchmark_sample_indices'],np.flatnonzero(held))
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN;tree=cKDTree(target)
        base=ROOT/('results/0929_baseline' if date=='0929' else f'results/multidate/{date}')
        baseline=np.array(json.loads((base/'alignment.json').read_text())['ply_to_las_local'])
        fixed=cKDTree(target[:,:2]).query(apply(benchmark[held],baseline)[:,:2])[0]<1.
        np.testing.assert_array_equal(data['fixed_coverage'],fixed)
        found={}
        for side in ['source','target']:
            f=json.loads((folder/f'{side}_candidates.json').read_text());clouds=np.load(folder/f'{side}_candidate_points.npz')
            for c in f['candidates']:c['points']=(clouds[c['id']]-f['normalization_center'])*f['normalization_scale']
            found[side]=f
        source_desc=descriptors(found['source']);target_desc=descriptors(found['target'])
        evaluated=0
        for run in a['runs']+[{'method':'existing_baseline','matrix_local':baseline.tolist()}]:
            if 'matrix_local' not in run:continue
            matrix=np.array(run['matrix_local']);linear=matrix[:3,:3];scale=np.cbrt(np.linalg.det(linear));assert scale>0
            np.testing.assert_allclose(linear.T@linear,np.eye(3)*scale**2,atol=1e-8)
            np.testing.assert_allclose(matrix[3],[0,0,0,1],atol=1e-12)
            errors=tree.query(apply(benchmark[held],matrix))[0]
            np.testing.assert_allclose(errors,data[run['method']],atol=1e-9)
            for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed)]:
                expected=next(r for r in a['comparison'] if r['method']==run['method'] and r['scope']==scope)
                for key,value in summary(errors[mask]).items():np.testing.assert_allclose(value,expected[key],atol=1e-9)
            evaluated+=1
            if run['method']=='existing_baseline':continue
            pairs=patch_checks(run['selected_initial_pairs'],found['source'],found['target'],matrix)
            for pair,saved in zip(pairs,run['final_pair_checks']):
                assert (pair['source_id'],pair['target_id'])==(saved['source_id'],saved['target_id'])
                for key in ['median_m','p90_m','fraction_within_1m','reverse_fraction_within_1m']:
                    np.testing.assert_allclose(pair[key],saved[key],atol=1e-9)
            geometric={(p['source_id'],p['target_id']) for p in match(source_desc,target_desc,matrix)}
            valid=[p for p in pairs if min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5 and (p['source_id'],p['target_id']) in geometric]
            support=spatial_support(valid,source_desc,target_desc,matrix)
            assert [(p['source_id'],p['target_id']) for p in support]==[(p['source_id'],p['target_id']) for p in run['final_spatial_support']]
            assert len(spatial_support([p for p in valid if p['fraction_within_1m']>=.8],source_desc,target_desc,matrix))==run['strict_forward_080_spatial_support_count']
        v=vertices[5::10];p=np.c_[v['x'],v['y'],v['z']].astype(float)
        p=p[(v['opacity']>0)&np.isfinite(p).all(1)];static_held=np.floor(p/.4).sum(1).astype(int)%4==0
        frozen=apply(p,baseline);saved=np.load(folder/'static_control_source_points.npz')
        for region in regions:
            reference=target[volume_mask(target,region)]
            for scope,mask in [('unseen_records',np.ones(len(p),bool)),('spatial_holdout',static_held)]:
                selected=p[volume_mask(frozen,region)&mask];np.testing.assert_array_equal(saved[region['id']+'_'+scope],selected)
                for run in a['runs']+[{'method':'existing_baseline','matrix_local':baseline.tolist()}]:
                    if 'matrix_local' not in run:continue
                    row=next(r for r in static_rows if r['date']==date and r['region']==region['id'] and r['scope']==scope and r['method']==run['method'])
                    assert int(row['count'])==len(selected)
                    if not len(selected):continue
                    distance=cKDTree(reference).query(apply(selected,np.array(run['matrix_local'])))[0]
                    for key,value in summary(distance).items():np.testing.assert_allclose(value,float(row[key]),atol=1e-9)
        checks.append({'date':date,'evaluated_matrices':evaluated,'training_records_checked':len(train),
            'withheld_records_checked':int(held.sum()),'static_records_checked':{key:len(saved[key]) for key in saved.files}})
        print('Verified',date,flush=True)
    for path,h in {**inputs,**preserved}.items():assert file_hash(ROOT/path)==h,path
    tests=(OUT/'tests.log').read_text();assert 'Ran 29 tests' in tests and tests.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':29,'original_input_hashes_matched':len(inputs),
        'registration_file_hashes_matched':len(preserved),'frozen_parameters_and_code_matched':True,
        'training_tile_leakage_count':0,'date_checks':checks,
        'meaning':'Numerical consistency and data preservation; not acceptance of every estimated pose.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('All checks passed.')


if __name__=='__main__':main()
