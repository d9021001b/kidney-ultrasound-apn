import unittest

import numpy as np
import pandas as pd

from nckuh_dmsa_native_saab_20261102 import G,L,summarize_kidneys
from nckuh_dmsa_cluster_ci import cluster_intervals


class NativeSaabTests(unittest.TestCase):
    def test_native_windows_never_cross_mask(self):
        mask=np.zeros((24,24),np.uint8); mask[2:22,2:22]=1; mask[10,10]=0
        gray=np.arange(24*24,dtype=np.uint8).reshape(24,24)
        for k in [5,9]:
            values,coords,total=L.get_patches(gray,mask,k,'test')
            for i,(y,x) in enumerate(coords):
                self.assertTrue(mask[y:y+k,x:x+k].all())
                np.testing.assert_allclose(values[i],gray[y:y+k,x:x+k].ravel()/255,atol=1e-7)

    def test_patient_balanced_sampler_excludes_other_patient(self):
        frame=pd.DataFrame(dict(patient_id=['train','train','heldout'],image_id=['a','b','c']))
        cache={5:{'a':np.ones((10,25),np.float32),'b':np.ones((10,25),np.float32)*2,'c':np.ones((10,25),np.float32)*99}}
        sampled,audit=L.training_patches(cache,frame,['train'],1,5)
        self.assertEqual(sampled.shape,(512,25)); self.assertLessEqual(sampled.max(),2)
        self.assertEqual({x['image_id'] for x in audit},{'a','b'})

    def test_existing_basis_orthonormal_and_response_dimensions(self):
        values=np.random.default_rng(1).random((200,25)).astype(np.float32)
        unit=G.fit_saab_unit(values)
        np.testing.assert_allclose(unit.W.T@unit.W,np.eye(25),atol=1e-6)
        self.assertEqual(L.local_statistics(values,unit,5).shape,(125,))

    def test_aggregation_never_mixes_kidneys(self):
        images=pd.DataFrame(dict(kidney_alias=['L','L','R','R']))
        kidneys=pd.DataFrame(dict(kidney_alias=['L','R']))
        got=summarize_kidneys(np.array([[1],[3],[20],[40]]),images,kidneys)
        np.testing.assert_equal(got,[[2,1],[30,10]])

    def test_cluster_ci_accepts_margins_and_links_two_sides(self):
        rows=[]
        for alias,code,ys in [('a','R',[0,1]),('b','L',[1,0]),('c','B',[1,1]),('d','N',[0,0])]:
            for side,y in enumerate(ys):
                for group in ['svm','reference']:
                    rows.append(dict(patient_alias=alias,kidney_alias=alias+str(side),DMSA_APN=code,label=y,
                        score=(2*y-1)*2,seed=1,group=group))
        intervals,differences,samples=cluster_intervals(pd.DataFrame(rows),['svm','reference'],[1],'reference',replicates=100)
        self.assertEqual(samples.shape,(100,2,3)); self.assertTrue(np.allclose(intervals['mean'],1))
        self.assertTrue(np.allclose(differences.difference,0))


if __name__=='__main__':
    unittest.main()
