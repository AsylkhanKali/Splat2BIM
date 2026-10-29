import unittest

import numpy as np

from semantic_projection import rasterize
from semantic_structures import LOVEDA_GROUP_IDS,context_projection,correspondence_bank
from align_object_recovery import surviving_pairs


class StructureTests(unittest.TestCase):
    def test_loveda_class_mapping(self):
        self.assertEqual(sorted(i for group in LOVEDA_GROUP_IDS for i in group),list(range(8)))
        self.assertEqual(LOVEDA_GROUP_IDS[1],[2])
        self.assertEqual(LOVEDA_GROUP_IDS[6],[])

    def test_context_does_not_invent_or_move_3d_samples(self):
        rng=np.random.default_rng(3);points=rng.normal(size=(500,3));colors=rng.random((500,3))
        _,old,_=rasterize(points,colors,[0,0,1]);image,new,metadata=context_projection(points,colors,[0,0,1])
        np.testing.assert_array_equal(old,new)
        self.assertEqual(image.shape,(1024,1024,3));self.assertGreater(metadata['context_only_filled_pixels'],0)
        self.assertTrue(np.all(np.isin(new[new>=0],np.arange(500))))

    def test_neighbor_bank_keeps_proposed_identities_and_excludes_far_starts(self):
        source={'ids':['a','b','c'],'centers':np.array([[0.,0,0],[20,0,0],[40,5,0]]),
            'lengths':np.ones((3,2))*5,'normals':np.tile([0,0,1.],(3,1))}
        target={**source,'ids':['x','y','z'],'centers':source['centers']+[4.9,0,0]}
        matrices=np.repeat(np.eye(4)[None],3,axis=0);matrices[:,0,3]=[-.2,.2,10]
        bank,neighbors=correspondence_bank(matrices,0,source['centers'],source,target)
        self.assertEqual(neighbors,[0,1]);self.assertEqual(len(bank),3)
        self.assertTrue(all(p['proposed_by_pool_indices']==[1] for p in bank))
        self.assertEqual(len(surviving_pairs(bank,source,target,matrices[0])),0)
        self.assertEqual(len(surviving_pairs(bank,source,target,matrices[1])),3)
        self.assertEqual(len({p['target_id'] for p in bank}),3)


if __name__=='__main__':unittest.main()
