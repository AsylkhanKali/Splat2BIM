import unittest

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from guarded_structural_pose import fractional_correction,global_guard,refine_with_context
from structural_pose_refinement import target_surface,extent_anchors


class GuardedStructuralTests(unittest.TestCase):
    def test_guard_rejects_tail_regression_even_when_median_improves(self):
        self.assertFalse(global_guard({'median':.2,'p90':.8},{'median':.19,'p90':.81}))
        self.assertTrue(global_guard({'median':.2,'p90':.8},{'median':.19,'p90':.79}))

    def test_fractional_update_preserves_scale_and_endpoints(self):
        initial=np.eye(4);initial[:3,:3]*=30;initial[:3,3]=[10,20,30]
        delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec([.001,-.002,.003]).as_matrix();delta[:3,3]=[.1,.2,.3]
        proposal=delta@initial;anchor=np.array([100.,50.,10.])
        np.testing.assert_allclose(fractional_correction(initial,proposal,0,anchor),initial,atol=1e-12)
        np.testing.assert_allclose(fractional_correction(initial,proposal,1,anchor),proposal,atol=1e-12)
        middle=fractional_correction(initial,proposal,.5,anchor)
        np.testing.assert_allclose(middle[:3,:3].T@middle[:3,:3],900*np.eye(3),atol=1e-10)

    def test_context_refinement_recovers_known_rigid_translation(self):
        x,y=np.meshgrid(np.linspace(-2,2,17),np.linspace(-2,2,17));p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        groups=[{'fit':p+center,'target':target_surface(p+center)} for center in [[0,0,0],[20,0,2],[0,20,4]]]
        points=np.concatenate([g['fit'] for g in groups]);context={'fit':points,'target':{'points':points,'tree':cKDTree(points)}}
        initial=np.eye(4);initial[:3,3]=[.03,-.04,.12]
        fitted,diagnostic=refine_with_context(groups,context,initial,extent_anchors(points))
        np.testing.assert_allclose(fitted,np.eye(4),atol=1e-6)
        self.assertTrue(np.all(np.diff(diagnostic['history'])<0))


if __name__=='__main__':unittest.main()
