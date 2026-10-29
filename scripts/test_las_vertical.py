import unittest
import numpy as np
from check_las_vertical import paired_heights,residual_metrics,symmetric_distance


class LasVerticalTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.arange(0,10,.25),np.arange(0,10,.25))
        self.roof=np.c_[x.ravel(),y.ravel(),10+.02*x.ravel()]

    def test_known_vertical_shift_and_reverse_sign(self):
        target=self.roof+[0,0,.7]
        fit,blocks,_=paired_heights(self.roof,target)
        reverse,_,_=paired_heights(target,self.roof)
        self.assertAlmostEqual(fit['delta_z_m'],.7)
        self.assertAlmostEqual(reverse['delta_z_m'],-.7)
        self.assertLess(symmetric_distance(self.roof,target,.7)['mean_directional_medians_m'],1e-10)
        self.assertGreaterEqual(len(blocks),5)

    def test_wrong_region_shift_fails_on_unchanged_control(self):
        fit,_,_=paired_heights(self.roof,self.roof+[0,0,.7])
        _,blocks,_=paired_heights(self.roof,self.roof)
        offsets=[r['delta_z_m'] for r in blocks]
        self.assertEqual(residual_metrics(offsets)['median_m'],0)
        self.assertAlmostEqual(residual_metrics(offsets,fit['delta_z_m'])['median_m'],.7)

    def test_missing_xy_overlap_is_not_zero_error_and_z_is_not_gated(self):
        fit,blocks,pairs=paired_heights(self.roof,self.roof+[100,100,0])
        self.assertIsNone(fit['delta_z_m']);self.assertEqual(len(blocks),0)
        fit,_,pairs=paired_heights(self.roof,self.roof+[0,0,12])
        self.assertAlmostEqual(fit['delta_z_m'],12)
        self.assertEqual(len(pairs['source']),len(self.roof))


if __name__=='__main__':unittest.main()
