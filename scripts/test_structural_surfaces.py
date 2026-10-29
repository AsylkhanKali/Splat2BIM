import unittest

import numpy as np

from structural_surfaces import grow_surfaces
from structural_correspondence import family_support,gated_pairs
from object_consensus import descriptors
from object_candidates import describe_cluster
from surface_infill import infill_surfaces
from align_stable_structures import surviving_bank


class StructuralSurfaceTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.arange(-4,4.01,.15),np.arange(-3,3.01,.15))
        self.roof=np.c_[x.ravel(),y.ravel(),np.full(x.size,2.)]
        self.seed=self.roof[(abs(self.roof[:,0])<1.5)&(abs(self.roof[:,1])<1)]

    def found(self,seeds):
        return {'normalization_center':np.zeros(3),'normalization_scale':1.,'ground_plane_normalized':np.array([0.,0,1,0]),
            'candidates':[{'id':name,'points':points} for name,points in seeds]}

    def test_grows_entire_connected_plane_without_disconnected_or_parallel_surface(self):
        points=np.r_[self.roof,self.roof+[20,0,0],self.roof+[0,0,1]]
        labels={'points':points,'labels':np.zeros(len(points),int)}
        grown=grow_surfaces(points,labels,self.found([('p0_c0',self.seed)]))
        self.assertEqual(len(grown['candidates']),1);q=grown['candidates'][0]['points']
        self.assertGreater(len(q),len(self.seed)*4)
        np.testing.assert_allclose(q[:,2],2.)
        self.assertLess(q[:,0].max(),4.1);self.assertGreater(np.ptp(q[:,0]),7.5)

    def test_fragments_and_overlapping_plane_slices_cannot_count_twice(self):
        labels={'points':self.roof,'labels':np.zeros(len(self.roof),int)}
        grown=grow_surfaces(self.roof,labels,self.found([('p0_c0',self.seed),('p0_c1',self.seed+[1,0,0]),('p1_c0',self.seed)]))
        self.assertEqual(len(grown['candidates']),1)
        self.assertEqual(grown['candidates'][0]['seed_ids'],['p0_c0','p0_c1','p1_c0'])

    def test_confident_movable_points_do_not_become_structural_support(self):
        labels={'points':self.roof,'labels':np.full(len(self.roof),6)}
        grown=grow_surfaces(self.roof,labels,self.found([('p0_c0',self.seed)]))
        self.assertEqual(grown['candidates'],[])

    def test_infill_uses_measured_points_inside_the_original_footprint(self):
        sparse=self.roof[::3];ground=np.array([0.,0,1,0])
        seed={**describe_cluster(sparse,ground),'id':'p0_c0','points':sparse}
        found=self.found([]);found['candidates']=[seed]
        points=np.r_[self.roof,self.roof+[0,0,1],self.roof+[20,0,0]]
        labels={'points':points,'labels':np.zeros(len(points),int)}
        rows=infill_surfaces(points,labels,found)
        self.assertEqual(len(rows),1);q=rows[0]['points']
        self.assertGreater(len(q),len(sparse))
        local=(q-seed['center'])@seed['frame']
        self.assertTrue(np.all(abs(local[:,:2])<=seed['lengths']/2+1e-9))
        np.testing.assert_allclose(q[:,2],2.)

    def test_one_surface_family_cannot_provide_two_spatial_supports(self):
        centers=[[0,0,2],[30,0,2],[0,25,2],[50,25,2]]
        found={'normalization_center':np.zeros(3),'normalization_scale':1.,'candidates':[
            {'id':str(i),'family':f,'center':np.array(c,float),'lengths':np.array([6.,4.]),'frame':np.eye(3)}
            for i,(c,f) in enumerate(zip(centers,['same','same','b','c']))]}
        pairs=gated_pairs(descriptors(found),descriptors(found),np.eye(4))
        support=family_support(pairs,found,found,np.eye(4))
        self.assertEqual(len(support),3)
        self.assertEqual(len({found['candidates'][p['source_index']]['family'] for p in support}),3)

    def test_all_geometrically_valid_alternatives_are_kept_but_gate_is_unchanged(self):
        def cloud(centers):
            return {'ids':[str(i) for i in range(len(centers))],'centers':np.array(centers,float),
                'lengths':np.tile([6.,4.],(len(centers),1)),'normals':np.tile([0.,0,1],(len(centers),1))}
        pairs=gated_pairs(cloud([[0,0,0]]),cloud([[1,0,0],[4.9,0,0],[5.1,0,0]]),np.eye(4))
        self.assertEqual([p['target_index'] for p in pairs],[0,1])

    def test_final_matching_cannot_add_an_identity_missing_before_icp(self):
        found={'normalization_center':np.zeros(3),'normalization_scale':1.,'candidates':[
            {'id':str(i),'family':str(i),'center':np.array(c,float),'lengths':np.array([6.,4.]),'frame':np.eye(3)}
            for i,c in enumerate([[0,0,2],[30,0,2],[0,25,2]])]}
        desc=descriptors(found);pairs=gated_pairs(desc,desc,np.eye(4))
        kept=surviving_bank(pairs[:1],desc,desc,np.eye(4))
        self.assertEqual(len(kept),1);self.assertEqual(kept[0]['source_id'],'0')

    def test_two_metre_second_axis_gate_is_not_relaxed(self):
        for second,expected in [(1.99,0),(2.01,3)]:
            centers=[[0,0,2],[20,second*np.sqrt(1.5),2],[40,0,2]]
            found={'normalization_center':np.zeros(3),'normalization_scale':1.,'candidates':[
                {'id':str(i),'family':str(i),'center':np.array(c,float),'lengths':np.array([6.,4.]),'frame':np.eye(3)}
                for i,c in enumerate(centers)]}
            pairs=gated_pairs(descriptors(found),descriptors(found),np.eye(4))
            self.assertEqual(len(family_support(pairs,found,found,np.eye(4))),expected)


if __name__=='__main__':unittest.main()
