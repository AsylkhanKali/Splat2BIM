import unittest

import numpy as np

from align_0929 import apply
from structural_pose_refinement import refine,target_surface,extent_anchors,select_candidate


class StructuralRefinementTests(unittest.TestCase):
    def groups(self):
        x,y=np.meshgrid(np.linspace(-2,2,17),np.linspace(-2,2,17));points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        return [{'family':str(i),'fit':points+center,'calibration':points[::3]+center,'target':target_surface(points+center)}
            for i,center in enumerate([[0,0,0],[20,0,2],[0,20,4]])]

    def test_rigid_update_recovers_small_shift_without_changing_scale(self):
        groups=self.groups();initial=np.eye(4);initial[:3,3]=[.03,-.04,.12]
        points=np.concatenate([g['fit'] for g in groups]);matrix,info=refine(groups,initial,1.,extent_anchors(points))
        np.testing.assert_allclose(matrix,np.eye(4),atol=1e-6)
        np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-12)
        self.assertTrue(np.all(np.diff(info['history'])<0))

    def test_parallel_planes_do_not_invent_horizontal_correction(self):
        groups=self.groups();initial=np.eye(4);initial[:3,3]=[.03,-.04,.12]
        points=np.concatenate([g['fit'] for g in groups]);matrix,_=refine(groups,initial,0.,extent_anchors(points))
        np.testing.assert_allclose(matrix[:2,3],[.03,-.04],atol=1e-8)
        self.assertLess(abs(matrix[2,3]),1e-8)

    def test_motion_limit_rejects_large_unjustified_displacement(self):
        groups=self.groups();initial=np.eye(4);initial[2,3]=.8
        points=np.concatenate([g['fit'] for g in groups]);anchors=extent_anchors(points)
        matrix,info=refine(groups,initial,0.,anchors)
        self.assertLessEqual(np.linalg.norm(apply(anchors,matrix)-apply(anchors,initial),axis=1).max(),.5+1e-12)

    def test_selection_rejects_strict_failure_and_family_regression(self):
        baseline={'loss':.04,'families':[{'median_m':.2}]}
        def candidate(loss,median,strict=True):
            return {'strict_pass':strict,'global_training_guard':True,'calibration':{'loss':loss,'families':[{'median_m':median}]}}
        rows=[candidate(.01,.1,False),candidate(.02,.23),candidate(.03,.19)]
        self.assertEqual(select_candidate(rows,baseline),2)
        self.assertIsNone(select_candidate(rows[:2],baseline))


if __name__=='__main__':unittest.main()
