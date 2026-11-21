import numpy as np

import nckuh_dmsa_fullframe_demographic_hierarchy_20261102 as study


def test_two_stage_marginals_preserve_expected_count():
    unilateral=np.array([.2,.7,.4])
    bilateral=np.array([.8,.1,.5])
    left_given=np.array([.6,.9,.2])
    left,right=study.compose_count_marginals(unilateral,bilateral,left_given)
    expected=unilateral+2*(1-unilateral)*bilateral
    np.testing.assert_allclose(left+right,expected)
    assert np.all((left>=0)&(left<=1)&(right>=0)&(right<=1))
