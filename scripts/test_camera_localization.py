import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_localization import project,local_pose,attach_depth,estimate_pose
from run_camera_localization import split_images


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[700.,0,600],[0,710,450],[0,0,1]])
        self.d=np.array([.07,-.05,.001,-.002,.01,.003,-.001,.001])

    def test_projection_matches_opencv_rational_camera(self):
        rng=np.random.default_rng(29);p=rng.uniform([-3,-3,10],[3,3,20],(100,3))
        r=Rotation.from_euler('xyz',[8,12,-9],degrees=True).as_matrix();t=np.array([2.,-1,3])
        actual,z=project(p,r,t,self.K,self.d)
        expected=cv2.projectPoints(p,cv2.Rodrigues(r)[0],t,self.K,self.d)[0].reshape(-1,2)
        np.testing.assert_allclose(actual,expected,atol=1e-9);self.assertTrue(np.all(z>0))

    def test_camera_pose_conversion_preserves_projection_under_sim3(self):
        r=Rotation.from_euler('xyz',[8,12,-9],degrees=True).as_matrix();t=np.array([2.,-1,3])
        transform=np.eye(4);transform[:3,:3]=25*Rotation.from_euler('xyz',[21,-9,30],degrees=True).as_matrix();transform[:3,3]=[100,-80,5]
        points=np.array([[0.,0,10],[3,1,12],[-2,1,15]])
        local_r,local_t=local_pose({'R':r,'t':t},transform)
        expected=project(points,r,t,self.K,self.d)[0]
        actual=project(points@transform[:3,:3].T+transform[:3,3],local_r,local_t,self.K,self.d)[0]
        np.testing.assert_allclose(actual,expected,atol=1e-9)

    def test_front_surface_wins_depth_lookup(self):
        x,y=np.meshgrid(np.arange(595,606),np.arange(445,456));xy=np.c_[x.ravel(),y.ravel()]
        rays=np.c_[(xy-[600,450])/[700,710],np.ones(len(xy))]
        p=np.r_[rays*10,rays*20]
        keep,points=attach_depth(np.array([[600.,450.]]),p,np.eye(3),np.zeros(3),self.K,np.zeros(8),(1200,900))
        self.assertTrue(keep[0]);np.testing.assert_allclose(points,[[0,0,10]],atol=1e-8)

    def test_pose_recovery_with_outliers_and_unused_image_blocks(self):
        rng=np.random.default_rng(29);points=rng.uniform([-8,-6,15],[8,6,30],(600,3))
        r=Rotation.from_euler('xyz',[3,-5,8],degrees=True).as_matrix();t=np.array([1.,-2,1])
        xy,z=project(points,r,t,self.K,self.d);xy+=rng.normal(0,.15,xy.shape)
        bad=rng.choice(len(points),120,replace=False);xy[bad]=rng.uniform([100,100],[1100,800],(len(bad),2))
        result,check=estimate_pose(xy,points,self.K,self.d,(1200,900))
        self.assertEqual(result['status'],'candidate_for_review')
        np.testing.assert_allclose(result['camera_center_local_m'],-r.T@t,atol=.03)
        self.assertTrue(np.all(~check['held'][check['inliers']]))

    def test_uncovered_features_do_not_acquire_depth(self):
        x,y=np.meshgrid(np.arange(595,606),np.arange(445,456));xy=np.c_[x.ravel(),y.ravel()]
        points=np.c_[(xy-[600,450])/[700,710],np.ones(len(xy))]*10
        keep,world=attach_depth(np.array([[100.,100.]]),points,np.eye(3),np.zeros(3),self.K,np.zeros(8),(1200,900))
        self.assertFalse(keep.any());self.assertEqual(world.shape,(0,3))

    def test_missing_correspondences_dont_produce_pose(self):
        result,check=estimate_pose(np.empty((0,2)),np.empty((0,3)),self.K,self.d,(1200,900))
        self.assertEqual(result['status'],'rejected_insufficient_correspondences');self.assertIsNone(check)

    def test_query_images_and_neighbours_are_excluded_from_references(self):
        names=[f'{i:04}' for i in range(1600)];q,r=split_images(names)
        self.assertFalse(set(q)&set(r));self.assertGreater(min(abs(int(a)-int(b)) for a in q for b in r),10)


if __name__=='__main__':unittest.main()
