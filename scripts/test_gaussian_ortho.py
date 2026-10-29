import unittest

import numpy as np

from gaussian_ortho import doubled_metadata,training_context,project_gaussians,render


class GaussianOrthoTests(unittest.TestCase):
    def test_training_context_excludes_holdouts_and_control_records(self):
        v=np.zeros(8,dtype=[(k,'f4') for k in ['x','y','z','opacity']]);v['x']=[.5,.5,.1,.5,.9,.5,1.3,.5]
        v['opacity']=0;v['opacity'][4]=-10
        _,ids=training_context(v,np.array([[0,0,0],[1,1,1]]))
        np.testing.assert_array_equal(ids,[0])

    def test_resolution_preserves_outer_pixel_bounds(self):
        old={'origin_xy':[5.,8.],'pixel_size_source_units':2.}
        new=doubled_metadata(old);local=np.array([[4.01,7.01],[2051.99,2054.99]])
        original_uv=(local-old['origin_xy'])/2;new_uv=(local-new['origin_xy'])/new['pixel_size_source_units']
        np.testing.assert_allclose(new_uv,2*original_uv+.5)
        self.assertTrue(np.all((np.rint(new_uv)>=0)&(np.rint(new_uv)<2048)))

    def test_alpha_composition_uses_depth_order(self):
        back=[4,4,1,0,1,3,3,.5,0,0,1];front=[4,4,1,0,1,3,3,.5,1,0,0]
        rgb,alpha=render(np.array([back,front]),size=9)
        np.testing.assert_allclose(rgb[4,4],[159,31,95],atol=1)
        self.assertAlmostEqual(float(alpha[4,4]),.75)
        reverse,_=render(np.array([front,back]),size=9)
        self.assertGreater(rgb[4,4,0],reverse[4,4,0])
        self.assertEqual(float(alpha[0,0]),0.)

    def test_projection_respects_covariance_and_sh(self):
        names=['x','y','z','opacity']+[f'rot_{i}' for i in range(4)]+[f'scale_{i}' for i in range(3)]+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]
        v=np.zeros(2,dtype=[(k,'f4') for k in names]);v['rot_0']=1;v['scale_0']=np.log(2);v['scale_1']=np.log(3);v['scale_2']=0;v['z']=[2,1];v['x']=[10,20];v['f_rest_1']=.5
        metadata={'frame':np.eye(3).tolist(),'pixel_size_source_units':1.,'center':[0,0,0],'origin_xy':[0,0]}
        rows=project_gaussians(v,metadata,size=64)
        np.testing.assert_allclose(rows[:,0],[20,10]);np.testing.assert_allclose(rows[:,1],63)
        np.testing.assert_allclose(rows[:,2],1/4.3,rtol=1e-6);np.testing.assert_allclose(rows[:,4],1/9.3,rtol=1e-6)
        np.testing.assert_allclose(rows[:,8],.5-.5*np.sqrt(3/(4*np.pi)),atol=1e-7)
        np.testing.assert_allclose(rows[:,9:],np.full((2,2),.5))


if __name__=='__main__':unittest.main()
