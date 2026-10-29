import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from align_0929 import apply
from pair_distance_seeds import from_pair,plane_frame,hypotheses
from align_object_recovery import eligible_candidates,surviving_pairs,scale_admissible


class PairDistanceTests(unittest.TestCase):
    def test_recovers_scale_rotation_and_translation_from_spaced_centers(self):
        source=np.array([[2.,3,1],[32,9,1]])
        rotation=Rotation.from_euler('xyz',[19,-28,37],degrees=True).as_matrix()
        target=source@(3.7*rotation).T+[23,-11,7]
        sf=plane_frame(np.tile([0.,0,1],(4,1)));tf=plane_frame(np.tile(rotation[:,2],(4,1)))
        fitted=from_pair(source,target,sf,tf)
        probes=np.array([[0,0,0],[3,5,9],[-1,5,2]])
        np.testing.assert_allclose(apply(probes,fitted),probes@(3.7*rotation).T+[23,-11,7],atol=1e-10)
        self.assertGreater(np.linalg.det(fitted[:3,:3]),0)

    def test_vertical_or_duplicate_centers_do_not_invent_yaw(self):
        for source in [np.array([[0.,0,0],[0,0,0]]),np.array([[0.,0,0],[0,0,4]])]:
            self.assertIsNone(from_pair(source,np.array([[0.,0,0],[30,0,0]]),np.eye(3),np.eye(3)))

    def test_constellation_survives_different_scene_extents_and_distractors(self):
        centers=np.array([[0.,0,0],[30,0,0],[0,25,0],[28,33,1]])
        rotation=Rotation.from_euler('xyz',[17,-9,29],degrees=True).as_matrix();scale=2.5
        target=centers@(scale*rotation).T+[17,-8,3]
        def found(points,normals,size,norm_scale):
            frame=plane_frame(np.tile(normals,(4,1)))
            return {'normalization_center':np.zeros(3),'normalization_scale':norm_scale,
                'candidates':[{'id':str(i),'center':p*norm_scale,'lengths':np.array([6.,4.])*size*norm_scale,
                    'frame':frame,'area':24*size**2*norm_scale**2} for i,p in enumerate(points)]}
        a=found(np.r_[centers,[[250,-300,3]]],[0.,0,1],1.,.1)
        b=found(np.r_[target,[[800,400,5]]],rotation[:,2],scale,.009)
        rows,_=hypotheses(a,b)
        self.assertTrue(rows)
        self.assertLess(min(np.max(np.linalg.norm(apply(centers,r['matrix'])-target,axis=1)) for r in rows),1e-8)

    def test_lower_geometric_error_cannot_override_missing_pair_support(self):
        candidates=[{'coarse_support_count':0,'training_coarse_median_m':.1},
            {'coarse_support_count':3,'training_coarse_median_m':.3},
            {'coarse_support_count':2,'training_coarse_median_m':.2}]
        eligible=eligible_candidates(candidates)
        self.assertEqual(eligible,[1])
        self.assertEqual(eligible_candidates(candidates[:1]),[])

    def test_scale_band_rejects_collapse_but_allows_large_extent_error(self):
        matrix=np.eye(4);matrix[:3,:3]*=22.
        self.assertTrue(scale_admissible(matrix,32.))
        matrix[:3,:3]=np.eye(3)*3.
        self.assertFalse(scale_admissible(matrix,32.))
        matrix[:3,:3]=np.eye(3)*70.
        self.assertFalse(scale_admissible(matrix,32.))

    def test_coarse_filter_cannot_swap_wrong_identity_for_nearby_target(self):
        source={'ids':['source'],'centers':np.array([[0.,0,0]]),'lengths':np.array([[6.,4.]]),'normals':np.array([[0.,0,1]])}
        target={'ids':['wrong','right'],'centers':np.array([[50.,0,0],[0,0,0]]),'lengths':np.array([[6.,4.],[6.,4.]]),'normals':np.array([[0.,0,1],[0,0,1]])}
        self.assertEqual(surviving_pairs([{'source_id':'source','target_id':'wrong'}],source,target,np.eye(4)),[])


if __name__=='__main__':unittest.main()
