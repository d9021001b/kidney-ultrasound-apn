import unittest
import numpy as np
import torch

from nckuh_dmsa_patch_mil_20261102 import PatchBagModel,balanced_patch_sample


class PatchMILTests(unittest.TestCase):
    def test_masked_padding_and_permutation_invariance(self):
        torch.manual_seed(1)
        x=torch.randn(2,6,32)
        mask=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,0]],dtype=torch.bool)
        order=torch.tensor([4,2,0,5,1,3])
        for mode in ['mean','gated']:
            model=PatchBagModel(mode).eval()
            with torch.inference_mode():
                original,attention=model(x,mask)
                permuted,_=model(x[:,order],mask[:,order])
                altered=x.clone(); altered[~mask]=100000
                padded,_=model(altered,mask)
            torch.testing.assert_close(original,permuted,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(original,padded,atol=1e-6,rtol=1e-6)
            self.assertTrue(torch.equal(attention[~mask],torch.zeros_like(attention[~mask])))
            torch.testing.assert_close(attention.sum(1),torch.ones(2))

    def test_balanced_sampler_never_uses_heldout_alias(self):
        aliases=np.array(['a']*70+['b']*20+['heldout']*100)
        selected=balanced_patch_sample(aliases,['a','b'],10)
        self.assertEqual(len(selected),128)
        self.assertEqual(dict(zip(*np.unique(aliases[selected],return_counts=True))),{'a':64,'b':64})

    def test_disclosed_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in PatchBagModel('mean').parameters()),2145)
        self.assertEqual(sum(p.numel() for p in PatchBagModel('gated').parameters()),2161)


if __name__=='__main__':
    unittest.main()
