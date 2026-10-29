import unittest
import numpy as np
from scipy.spatial import cKDTree

from align_0929 import apply
from semantic_matching import SemanticMatcher
from semantic_initial import score_pool,choose_starts


class InitialSemanticTests(unittest.TestCase):
    def test_batch_scores_equal_individual_class_matching(self):
        rng=np.random.default_rng(29);points=rng.normal(size=(400,3));labels=np.r_[np.ones(200),np.full(200,4)].astype(int)
        data={'points':points,'labels':labels};matcher=SemanticMatcher(data,data);tree=cKDTree(points)
        matrices=np.tile(np.eye(4),(140,1,1));matrices[:,:3,3]=rng.uniform(-5,5,(140,3))
        score=score_pool(points[:20],tree,matcher,matrices)
        for i in [0,127,128,139]:
            self.assertAlmostEqual(score[i,1],matcher.score(matrices[i])[0],places=10)
            self.assertAlmostEqual(score[i,0],np.quantile(tree.query(apply(points[:20],matrices[i]))[0],.75),places=10)

    def test_distinct_starts_share_budget_and_missing_semantics_abstains(self):
        matrices=np.tile(np.eye(4),(6,1,1));matrices[:,0,3]=[0,.1,3,6,9,12]
        scores=np.c_[np.arange(6.),np.zeros(6)];anchors=np.array([[0.,0,0],[1,2,3]])
        self.assertEqual(choose_starts(matrices,scores,anchors),[0,2,3,4])
        scores[:,1]=np.nan;self.assertEqual(choose_starts(matrices,scores,anchors,True),[])


if __name__=='__main__':unittest.main()
