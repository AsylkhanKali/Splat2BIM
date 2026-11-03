import unittest
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from registration_ablation import apply,similarity,pca_hypotheses,held_tiles,semantic_classes,scores,refine,matrix_parts,measures
from run_registration_ablation import configurations

class AblationTests(unittest.TestCase):
    def test_known_similarity_and_global_origin(self):
        x=np.random.default_rng(8).normal(size=(500,3))*[4,2,1]
        T=np.eye(4);T[:3,:3]=27*Rotation.from_euler('xyz',[.4,-.3,.2]).as_matrix();T[:3,3]=[8,4,-12]
        recovered=similarity(x,apply(x,T));np.testing.assert_allclose(recovered,T,atol=1e-10)
        global_T=T.copy();global_T[:3,3]+=[825800,842100,0]
        np.testing.assert_allclose(apply(x,global_T),apply(x,T)+[825800,842100,0])
        np.testing.assert_allclose(apply(apply(x,T),np.linalg.inv(T)),x,atol=1e-12)
    def test_pca_contains_true_anisotropic_pose(self):
        x=np.random.default_rng(9).normal(size=(1000,3))*[5,2,.3]
        T=np.eye(4);T[:3,:3]=12*Rotation.from_euler('xyz',[.3,.1,-.8]).as_matrix();T[:3,3]=[9,8,7]
        pool=pca_hypotheses(x,apply(x,T));self.assertEqual(len(pool),4)
        self.assertLess(min(np.max(abs(m-T)) for m in pool),1e-9)
        for m in pool:matrix_parts(m)
    def test_icp_recovers_small_known_pose(self):
        x=np.random.default_rng(5).normal(size=(2500,3))*[3,2,1]
        T=np.eye(4);T[:3,:3]=1.02*Rotation.from_euler('z',.02).as_matrix();T[:3,3]=[.04,.02,-.03]
        result,_,_=refine(x,apply(x,T),np.eye(4),iterations=40)
        np.testing.assert_allclose(result,T,atol=1e-7)
    def test_labels_change_ranking_not_coordinates(self):
        rng=np.random.default_rng(1);x=np.r_[rng.normal(scale=.1,size=(150,3)),rng.normal(scale=.1,size=(150,3))+[10,0,0]]
        a={'points':x.copy(),'labels':np.repeat([1,3],150),'confidence':np.ones(300)}
        b={**a,'labels':np.repeat([3,1],150)};before=x.copy();T=np.eye(4)
        good=semantic_classes(a,a,.6,29);bad=semantic_classes(a,b,.6,29)
        self.assertEqual(scores(T,x,cKDTree(x),good,.5)['combined'],0.)
        self.assertGreater(scores(T,x,cKDTree(x),bad,.5)['combined'],1.)
        np.testing.assert_array_equal(x,before);np.testing.assert_array_equal(T,np.eye(4))
    def test_missing_classes_rejected_not_zero_score(self):
        a={'points':np.zeros((10,3)),'labels':np.ones(10),'confidence':np.ones(10)}
        classes=semantic_classes(a,a,.6,29)
        self.assertIsNone(scores(np.eye(4),a['points'],cKDTree(a['points']),classes,.5)['combined'])
    def test_spatial_split_same_for_duplicate_records(self):
        x=np.random.default_rng(2).normal(size=(1000,3));tiles,held=held_tiles(x)
        np.testing.assert_array_equal(held,held_tiles(x.copy())[1]);self.assertTrue(held.any());self.assertTrue((~held).any())
        self.assertFalse(set(map(tuple,tiles[held]))&set(map(tuple,tiles[~held])))
    def test_metric_includes_large_errors(self):
        m=measures(np.r_[np.ones(99),1000.]);self.assertGreater(m['rmse_m'],99);self.assertEqual(m['n'],100)
    def test_zero_weight_semantics_is_geometry(self):
        x=np.eye(3);r=scores(np.eye(4),x,cKDTree(x),[],0.)
        self.assertEqual(r['combined'],r['geometry'])
    def test_matched_primary_budget(self):
        c=configurations();a,b=c[1:3]
        self.assertEqual({k:v for k,v in a.items() if k not in ('name','weight')},{k:v for k,v in b.items() if k not in ('name','weight')})

if __name__=='__main__':unittest.main()
