import numpy as np
import pandas as pd

import nckuh_dmsa_fullframe_image_instance_probe_20261102 as study


def test_balanced_image_weights_equalize_kidneys_and_classes():
    aliases=np.array(["a","a","b","c","c","c"])
    labels=np.array([0,0,0,1,1,1])
    weights=study.balanced_image_weights(aliases,labels)
    kidney_totals={alias:weights[aliases==alias].sum() for alias in np.unique(aliases)}
    assert kidney_totals["a"]==kidney_totals["b"]
    np.testing.assert_allclose(weights[labels==0].sum(),weights[labels==1].sum())


def test_pooling_is_one_score_per_kidney():
    images=pd.DataFrame({"kidney_alias":["k1","k1","k2"]})
    kidneys=pd.DataFrame({"kidney_alias":["k1","k2"]})
    score=np.array([.2,.8,.7])
    probability=study.pool_to_kidneys(score,images,kidneys,"probability_mean")
    logit=study.pool_to_kidneys(score,images,kidneys,"logit_mean")
    np.testing.assert_allclose(probability,[.5,.7])
    np.testing.assert_allclose(logit,[.5,.7])
