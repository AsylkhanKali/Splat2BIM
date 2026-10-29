import unittest
import numpy as np
from check_static_structures import volume_mask,measurements,mean_region_score,roof_plane


class StaticStructureTests(unittest.TestCase):
    def test_mask_excludes_ground_vehicle_and_overhead_crane(self):
        r={'polygon':[[0,0],[10,0],[10,10],[0,10]],'z_range':[8,16]}
        p=np.array([[5,5,10],[5,5,4],[5,5,45],[12,5,10],[0,0,8.]])
        np.testing.assert_array_equal(volume_mask(p,r),[True,False,False,False,True])
        r['polygon'].reverse()
        np.testing.assert_array_equal(volume_mask(p,r),[True,False,False,False,True])

    def test_large_distance_inside_mask_is_not_discarded(self):
        p=np.array([[0,0,10.],[0,0,15.]])
        errors,metrics=measurements(p,np.array([[0,0,10.]]))
        np.testing.assert_array_equal(errors,[0,5]);self.assertEqual(metrics['count'],2)
        self.assertEqual(metrics['median_m'],2.5)
        self.assertIsNone(mean_region_score([metrics]))

    def test_roof_height_at_same_anchor_recovers_known_difference(self):
        xy=np.random.default_rng(4).uniform(-10,10,(1000,2));z=.03*xy[:,0]-.05*xy[:,1]+11
        first=roof_plane(np.c_[xy,z],np.array([2.,3.]))
        second=roof_plane(np.c_[xy,z+.7],np.array([2.,3.]))
        self.assertAlmostEqual(first[2],10.91,places=8)
        self.assertAlmostEqual(second[2]-first[2],.7,places=8)


if __name__=='__main__':unittest.main()
