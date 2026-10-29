import unittest
import numpy as np

from semantic_projection import GROUP_IDS,rasterize,visible_labels
from semantic_matching import SemanticMatcher,choose_candidate


class SemanticTests(unittest.TestCase):
    def test_model_labels_partition_without_dropping_probability(self):
        self.assertEqual(sorted(i for group in GROUP_IDS for i in group),list(range(24)))

    def test_raster_keeps_frontmost_actual_point(self):
        x,y=np.meshgrid(np.arange(12.),np.arange(8.));ground=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        points=np.r_[ground,ground+[0,0,2]];colors=np.ones_like(points)*.5
        image,indices,metadata=rasterize(points,colors,[0,0,1],64)
        self.assertTrue(np.all(indices[indices>=0]>=len(ground)))
        self.assertEqual(metadata['visible_points'],len(ground))
        self.assertEqual(image.shape,(64,64,3))

    def test_missing_pixels_never_become_semantic_3d_points(self):
        indices=np.array([[0,-1],[-1,1]]);prob=np.zeros((2,2,7));prob[:,:,1]=.9;prob[:,:,0]=.1
        points=np.array([[1.,2,3],[4,5,6]])
        data=visible_labels(points,indices,prob)
        np.testing.assert_array_equal(data['points'],points);self.assertEqual(len(data['labels']),2)

    def test_low_confidence_is_unknown(self):
        data=visible_labels(np.array([[1.,2,3]]),np.array([[0]]),np.full((1,1,7),1/7))
        self.assertEqual(data['labels'].tolist(),[0])

    def test_semantics_disambiguates_equal_geometry(self):
        rng=np.random.default_rng(29);patch=rng.uniform([0,0,0],[1,1,1],(150,3))
        points=np.r_[patch,patch+[10,0,0]];labels=np.r_[np.ones(150),np.full(150,4)].astype(int)
        data={'points':points,'labels':labels};matcher=SemanticMatcher(data,data)
        wrong=np.eye(4);wrong[0,0]=-1;wrong[1,1]=-1;wrong[:3,3]=[11,1,0]
        correct_score,_=matcher.score(np.eye(4));wrong_score,_=matcher.score(wrong)
        self.assertLess(correct_score,.001);self.assertGreater(wrong_score,8)
        rows=[{'geometry_score_m':.1,'semantic_score_m':wrong_score,'geometric_support_count':3,'scale_admissible':True},
            {'geometry_score_m':.2,'semantic_score_m':correct_score,'geometric_support_count':3,'scale_admissible':True}]
        self.assertEqual(choose_candidate(rows),0);self.assertEqual(choose_candidate(rows,True),1)

    def test_one_class_is_insufficient_semantic_support(self):
        data={'points':np.zeros((200,3)),'labels':np.ones(200,int)}
        self.assertIsNone(SemanticMatcher(data,data).score(np.eye(4))[0])

    def test_semantics_cannot_override_geometric_or_scale_rejection(self):
        rows=[{'geometry_score_m':.01,'semantic_score_m':0.,'geometric_support_count':2,'scale_admissible':True},
            {'geometry_score_m':.01,'semantic_score_m':0.,'geometric_support_count':3,'scale_admissible':False}]
        self.assertIsNone(choose_candidate(rows,True));self.assertIsNone(choose_candidate(rows))


if __name__=='__main__':unittest.main()
