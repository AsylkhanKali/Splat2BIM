import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from align_0929 import apply
from check_las_rigid import rigid_fit,linear_system,target_surface,evaluate


class LasRigidTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.linspace(-5,5,41),np.linspace(-5,5,41))
        x=x.ravel();y=y.ravel();self.plane=np.c_[x,y,np.zeros(len(x))]
        self.targets={'floor':target_surface(self.plane),
            'walls':target_surface(np.r_[np.c_[np.full(len(x),-5),x,y+5],np.c_[x,np.full(len(x),5),y+5]])}

    def test_known_small_rigid_transform_on_nonparallel_surfaces(self):
        truth=np.eye(4);truth[:3,:3]=Rotation.from_euler('xyz',[.4,-.3,.2],degrees=True).as_matrix()
        truth[:3,3]=[.12,-.08,.18]
        source={k:apply(t['points'],np.linalg.inv(truth)) for k,t in self.targets.items()}
        fitted,info=rigid_fit(source,self.targets,np.eye(4),np.zeros(3))
        np.testing.assert_allclose(fitted,truth,atol=.015)
        self.assertEqual(info['rank_at_relative_1e_3'],6)
        for stage in info['iterations']:
            self.assertTrue(np.all(np.diff(stage['accepted_losses'])<=0))

    def test_plane_cannot_determine_inplane_shift_or_yaw(self):
        target={'roof':target_surface(self.plane)}
        jac,_,_,_=linear_system({'roof':self.plane},target,np.eye(4),np.zeros(3),.5)
        self.assertEqual(np.linalg.matrix_rank(jac),3)

    def test_control_retains_all_points_after_large_wrong_translation(self):
        moved=np.eye(4);moved[:3,3]=[0,0,12]
        score=evaluate(self.plane,self.plane,moved)
        self.assertEqual(score['source_count'],len(self.plane))
        self.assertAlmostEqual(score['symmetric_median_m'],12)
        self.assertAlmostEqual(score['source_p90_m'],12)


if __name__=='__main__':unittest.main()
