import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from object_candidates import rectangle,describe_cluster,guesses
from align_0929 import apply


class ObjectCandidateTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.linspace(-3,3,25),np.linspace(-2,2,21))
        self.roof=np.c_[x.ravel(),y.ravel(),np.full(x.size,1.)]

    def test_rectangle_dimensions_under_arbitrary_rotation(self):
        r=Rotation.from_euler('xyz',[24,-37,19],degrees=True).as_matrix()
        fit=rectangle(self.roof@r.T+[5,-3,8])
        np.testing.assert_allclose(fit['lengths'],[6,4],atol=1e-8)
        self.assertAlmostEqual(fit['rectangle_fill'],1)
        self.assertAlmostEqual(np.linalg.det(fit['frame']),1)

    def test_surface_rule_rejects_ground_and_high_thin_structure(self):
        ground=np.array([0,0,1,0])
        self.assertTrue(describe_cluster(self.roof,ground)['accepted'])
        self.assertFalse(describe_cluster(self.roof-[0,0,1],ground)['accepted'])
        self.assertFalse(describe_cluster(self.roof+[0,0,10],ground)['accepted'])

    def test_missing_objects_produce_no_pose_hypothesis(self):
        empty={'normalization_scale':1.,'candidates':[]}
        self.assertEqual(guesses(empty,empty),[])

    def test_object_hypotheses_include_known_proper_similarity(self):
        r=Rotation.from_euler('xyz',[24,-37,19],degrees=True).as_matrix();t=np.array([4.,-8.,3.]);scale=7.
        src=rectangle(self.roof);dst=rectangle(self.roof@r.T+t/scale)
        # Consistent upward normal, as in the ground-orientation detector.
        if dst['frame'][:,2]@r[:,2]<0:dst['frame'][:,1:]*=-1
        if src['frame'][2,2]<0:src['frame'][:,1:]*=-1
        source={'normalization_scale':1.,'normalization_center':np.zeros(3),'candidates':[dict(src,id='source')]}
        target={'normalization_scale':1/scale,'normalization_center':np.zeros(3),'candidates':[dict(dst,id='target')]}
        candidates=guesses(source,target);truth=self.roof@(scale*r).T+t
        errors=[np.max(np.linalg.norm(apply(self.roof,c['matrix'])-truth,axis=1)) for c in candidates]
        self.assertLess(min(errors),1e-8)
        self.assertTrue(all(np.linalg.det(c['matrix'][:3,:3])>0 for c in candidates))


if __name__=='__main__':unittest.main()
