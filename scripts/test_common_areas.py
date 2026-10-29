"""Check spatial comparisons at grid boundaries, missing coverage and unequal density."""
import unittest
import numpy as np
from check_common_areas import grid_indices,summarize_cells,common_cells,balanced_metrics


class CommonAreaTests(unittest.TestCase):
    def test_grid_uses_floor_and_half_open_cells(self):
        p=np.array([[-.01,-20,0],[0,19.999,0],[20,20,0],[-20.001,0,0]])
        np.testing.assert_array_equal(grid_indices(p),[[-1,-1],[0,0],[1,1],[-2,0]])

    def test_common_support_requires_every_date_and_minimum_count(self):
        rows={'a':{'0,0':{'count':50},'1,0':{'count':60},'2,0':{'count':70}},
              'b':{'0,0':{'count':50},'1,0':{'count':49}}}
        self.assertEqual(common_cells(rows),['0,0'])

    def test_equal_cell_weight_differs_from_dense_point_weight(self):
        p=np.r_[np.tile([1,1,0.],(1000,1)),np.tile([21,1,0.],(50,1))]
        e=np.r_[np.full(1000,.1),np.full(50,2.)]
        rows=summarize_cells(p,e)
        result=balanced_metrics(rows,['0,0','1,0'])
        self.assertAlmostEqual(result['median_of_cell_medians_m'],1.05)
        self.assertAlmostEqual(result['mean_cell_fraction_within_1m'],.5)
        self.assertEqual(result['samples'],1050)
        self.assertAlmostEqual(np.median(e),.1)


if __name__=='__main__':unittest.main()
