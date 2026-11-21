import numpy as np

import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as study


def test_swap_pairs_is_involution():
    values=np.arange(24).reshape(3,8)
    np.testing.assert_array_equal(study.swap_pairs(study.swap_pairs(values)),values)


def test_hierarchical_marginals_respect_expected_count():
    count=np.array([[.2,.3,.5],[.7,.2,.1]])
    left_given=np.array([.8,.25])
    left=count[:,2]+count[:,1]*left_given;right=count[:,2]+count[:,1]*(1-left_given)
    np.testing.assert_allclose(left+right,count[:,1]+2*count[:,2])
    assert np.all((left>=0)&(left<=1)&(right>=0)&(right<=1))


def test_powerset_symmetry_swaps_raw_pairs_before_preprocessing():
    class TrackingPreprocessor:
        def __init__(self):
            self.calls=[]

        def transform(self,values):
            self.calls.append(values.copy())
            return values[:,[0,2]]+2*values[:,[1,3]]

    class FixedModel:
        classes_=np.arange(4)

        def predict_proba(self,values):
            logits=np.c_[np.zeros(len(values)),values[:,0],values[:,1],values.sum(1)]
            logits-=logits.max(1,keepdims=True)
            probabilities=np.exp(logits)
            return probabilities/probabilities.sum(1,keepdims=True)

    raw=np.arange(16,dtype=float).reshape(4,4)
    preprocessor=TrackingPreprocessor()
    left,right=study.powerset_scores(FixedModel(),preprocessor,raw)
    np.testing.assert_array_equal(preprocessor.calls[0],raw)
    np.testing.assert_array_equal(preprocessor.calls[1],study.swap_pairs(raw))
    swapped_left,swapped_right=study.powerset_scores(FixedModel(),TrackingPreprocessor(),study.swap_pairs(raw))
    np.testing.assert_allclose(left,swapped_right)
    np.testing.assert_allclose(right,swapped_left)
