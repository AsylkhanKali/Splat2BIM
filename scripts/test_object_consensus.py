import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from align_0929 import apply
from object_consensus import match,spatial_support,hypotheses
from align_object_consensus import patch_checks


def objects(centers):
    return {'ids':[str(i) for i in range(len(centers))],'centers':np.array(centers,float),
        'lengths':np.tile([6.,4.],(len(centers),1)),'normals':np.tile([0.,0.,1.],(len(centers),1))}


class ConsensusTests(unittest.TestCase):
    def test_three_distributed_pairs_with_known_transform_and_distractors(self):
        source=objects([[0,0,0],[30,0,0],[0,25,0],[18,17,1]])
        matrix=np.eye(4);matrix[:3,:3]=2*Rotation.from_euler('z',38,degrees=True).as_matrix();matrix[:3,3]=[7,-9,4]
        target=objects(np.r_[apply(source['centers'],matrix),[[150,100,4]]]);target['lengths']*=2
        pairs=match(source,target,matrix)
        self.assertEqual(len(pairs),4)
        self.assertEqual(len(spatial_support(pairs,source,target,matrix)),4)

    def test_repeated_source_fragments_cannot_reuse_target(self):
        source=objects([[0,0,0],[1,0,0],[2,0,0]])
        target=objects([[0,0,0]])
        self.assertEqual(len(match(source,target,np.eye(4))),1)

    def test_collinear_or_nearby_matches_do_not_establish_spatial_support(self):
        for xyz in ([[0,0,0],[20,0,0],[40,0,0]],[[0,0,0],[1,0,0],[0,1,0]]):
            source=objects(xyz);pairs=match(source,source,np.eye(4))
            self.assertEqual(spatial_support(pairs,source,source,np.eye(4)),[])

    def test_wrong_height_dimensions_and_normal_are_rejected(self):
        source=objects([[0,0,0]])
        for mode in ['height','size','normal']:
            target=objects([[0,0,0]])
            if mode=='height':target['centers'][:,2]=6
            if mode=='size':target['lengths']*=3
            if mode=='normal':target['normals']*=-1
            self.assertEqual(match(source,target,np.eye(4)),[])

    def test_hypotheses_recover_distributed_constellation(self):
        centers=np.array([[0.,0,0],[30,0,0],[0,25,0],[18,17,1]])
        rotation=Rotation.from_euler('xyz',[23,-17,39],degrees=True).as_matrix()
        def found(c,r,s):
            return {'normalization_center':np.zeros(3),'normalization_scale':s,
                'candidates':[{'id':str(i),'center':point,'frame':r,'lengths':np.array([6.,4.]),'area':24.}
                    for i,point in enumerate(c)]}
        a=found(centers,np.eye(3),1.);b=found(centers@rotation.T+[7,-9,4],rotation,.5)
        _,rows=hypotheses(a,b)
        truth=(centers@rotation.T+[7,-9,4])*2
        self.assertTrue(rows)
        self.assertLess(min(np.max(np.linalg.norm(apply(centers,r['matrix'])-truth,axis=1)) for r in rows),1e-8)

    def test_patch_check_keeps_wrong_identity_and_measures_both_directions(self):
        points=np.c_[np.arange(10),np.zeros((10,2))]
        def found(rows):
            return {'normalization_scale':1.,'normalization_center':np.zeros(3),
                'candidates':[{'id':name,'points':p} for name,p in rows]}
        source=found([('source',points)]);target=found([('wrong',points+100),('right',points[:5])])
        rows=patch_checks([{'source_id':'source','target_id':'wrong'},
            {'source_id':'source','target_id':'right'}],source,target,np.eye(4))
        self.assertEqual(rows[0]['fraction_within_1m'],0)
        self.assertEqual(rows[0]['target_id'],'wrong')
        self.assertEqual(rows[1]['fraction_within_1m'],.5)
        self.assertEqual(rows[1]['reverse_fraction_within_1m'],1)


if __name__=='__main__':unittest.main()
