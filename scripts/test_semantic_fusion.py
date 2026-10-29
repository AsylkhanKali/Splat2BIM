import unittest
import numpy as np
from semantic_fusion import expanded_context,fuse

class FusionTests(unittest.TestCase):
    def test_building_expert_cannot_override_protected_classes(self):
        drone={'labels':np.array([0,1,2,3,4,5,6],np.uint8),'confidence':np.ones(7)*.8}
        expert=np.zeros((7,7));expert[:,1]=.9
        labels,confidence,origin=fuse(drone,expert)
        np.testing.assert_array_equal(labels,[1,1,1,1,4,5,6])
        np.testing.assert_array_equal(origin,[1,1,1,1,0,0,0])
        expert[:,1]=.3;labels,_,_=fuse(drone,expert)
        np.testing.assert_array_equal(labels,[0,0,2,3,4,5,6])

    def test_context_uses_only_observed_colors(self):
        points=np.array([[20.,20,0],[80,20,0],[20,80,0],[80,80,0]])
        metadata={'center':[0,0,0],'frame':np.eye(3).tolist(),'origin_xy':[0,0],'pixel_size_source_units':1.}
        image=np.full((100,100,3),255,np.uint8);image[99-points[:,1].astype(int),points[:,0].astype(int)]=[20,100,180]
        out,uv,radius=expanded_context(image,points,metadata)
        self.assertEqual(radius,24);self.assertEqual(len(uv),len(points))
        self.assertFalse(np.any(np.all(out==255,axis=2)))

if __name__=='__main__':unittest.main()
