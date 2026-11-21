import unittest
import numpy as np

from nckuh_dmsa_label_powerset_20261102 import CODE,SWAP,swap_pair,joint_marginals


class JointLabelTests(unittest.TestCase):
    def test_side_exchange_is_involution(self):
        x=np.arange(24).reshape(3,8)
        np.testing.assert_equal(swap_pair(swap_pair(x)),x)
        np.testing.assert_equal(SWAP[SWAP],np.arange(4))
        self.assertEqual(SWAP[CODE['L']],CODE['R'])
        self.assertEqual(SWAP[CODE['B']],CODE['B'])

    def test_joint_marginals_are_exchange_equivariant(self):
        class Model:
            classes_=np.array([3,1,0,2])
            def predict_proba(self,x):
                scores=np.c_[x[:,0]+x[:,2]+.3,x[:,0],np.zeros(len(x)),x[:,2]]
                values=np.exp(scores-scores.max(1,keepdims=True))
                return values/values.sum(1,keepdims=True)
        x=np.array([[.2,1.,.8,1.],[.9,1.,.3,1.]])
        first=joint_marginals(Model(),x)
        second=joint_marginals(Model(),swap_pair(x))
        np.testing.assert_allclose(first,second[:,::-1])
        self.assertTrue(((first>=0)&(first<=1)).all())


if __name__=='__main__':
    unittest.main()
