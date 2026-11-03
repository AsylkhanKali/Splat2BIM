"""Independently recompute frozen cohort metrics, ranks and transform conventions."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from project_data import ROOT,ORIGIN,file_hash
from registration_ablation import apply,voxel,sample,pca_hypotheses,semantic_classes,scores,matrix_parts,measures
from run_registration_ablation import load_data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/ablation');args=parser.parse_args();out=args.out
    protocol=json.loads((out/'protocol.json').read_text());results=json.loads((out/'results.json').read_text())
    for name,h in protocol['input_sha256'].items():assert file_hash(ROOT/name)==h,name
    for name,h in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==h,name
    train,evaluation,fit_ids,test_ids,tiles,target,source_labels,target_labels=load_data();saved=np.load(out/'evaluation.npz')
    for name,value in [('source',evaluation),('source_indices',test_ids),('fit_indices',fit_ids),('tiles',tiles)]:np.testing.assert_array_equal(saved[name],value)
    assert not np.intersect1d(fit_ids,test_ids).size
    pca_source=voxel(train,.06);coarse=voxel(target,2.);pool=pca_hypotheses(pca_source,coarse)
    np.testing.assert_allclose(np.load(out/'pca_candidates.npz')['matrices'],pool,atol=1e-10,rtol=0)
    expected={(c['name'],s) for c in protocol['configurations'] for s in protocol['seeds']}
    assert len(results)==len(expected)==len({(r['name'],r['seed']) for r in results})
    assert {(r['name'],r['seed']) for r in results}==expected
    tree=cKDTree(target);ranktree=cKDTree(coarse);checked=0
    for r in results:
        config=r['config'];seed=r['seed'];rank=sample(pca_source,4096,seed);folder=out/r['name']/str(seed)
        classes=semantic_classes(source_labels,target_labels,config['confidence'],seed,config['shuffle']) if config['weight'] else []
        initial=[scores(T,rank,ranktree,classes,config['weight']) for T in pool]
        assert initial==r['initial_scores']
        order=sorted((i for i in range(4) if initial[i]['combined'] is not None),key=lambda i:(initial[i]['combined'],i))[:config['starts']]
        assert order==r['selected_starts']
        if r['metrics'] is None:
            assert not r['candidates'];continue
        for c in r['candidates']:
            s=scores(np.array(c['matrix']),rank,ranktree,classes,config['weight']);assert s==c['training_score']
        selected=min(r['candidates'],key=lambda c:(c['training_score']['combined'],c['index']));assert selected['index']==r['selected_index']
        assert selected['matrix']==r['matrix'];T=np.array(r['matrix']);matrix_parts(T)
        np.testing.assert_allclose(apply(apply(evaluation[:100],T),np.linalg.inv(T)),evaluation[:100],atol=1e-9)
        for stage,metric in r['stages'].items():
            L=np.loadtxt(folder/f'T_{stage}_local.txt');G=np.loadtxt(folder/f'T_{stage}_global.txt')
            np.testing.assert_allclose(G[:3,3]-L[:3,3],ORIGIN,atol=1e-8,rtol=0)
            d=tree.query(apply(evaluation,L),workers=2)[0];recomputed=measures(d)
            for key in metric:np.testing.assert_allclose(metric[key],recomputed[key],atol=1e-9,rtol=0)
        np.testing.assert_array_equal(np.loadtxt(folder/'T_pca_local.txt'),np.loadtxt(folder/'T_after_semantic_labels_local.txt'))
        d=tree.query(apply(evaluation,T),workers=2)[0]
        np.testing.assert_allclose(d,np.load(folder/'errors.npz')['distance_m'],atol=1e-10,rtol=0)
        for key,v in measures(d).items():np.testing.assert_allclose(v,r['metrics'][key],atol=1e-10,rtol=0)
        checked+=1
    receipt={'status':'passed','date':'0929','expected_runs':len(expected),'evaluated_runs_verified':checked,'failed_runs_retained':len(results)-checked,
      'same_evaluation_indices':True,'fit_evaluation_overlap':0,'raw_input_and_runner_hashes_match':True,
      'pca_pool_recomputed':True,'training_only_candidate_ranking_recomputed':True,'all_stage_distances_recomputed':True,
      'semantic_label_stage_matrix_unchanged':True,'local_global_and_inverse_conventions_verified':True,
      'protocol_sha256':file_hash(out/'protocol.json'),'results_sha256':file_hash(out/'results.json'),'verifier_sha256':file_hash(Path(__file__)),
      'limits':'Numerical and reproducibility checks, not independent survey validation. Fits were not rerun by this verifier.'}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
