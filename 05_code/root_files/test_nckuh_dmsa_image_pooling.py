import numpy as np
import pandas as pd

from nckuh_dmsa_image_pooling import pool


def test_focal_pooling_order_and_alias_alignment():
    images = pd.DataFrame({"kidney_alias": ["A", "A", "B", "B"]})
    scores = np.asarray([.1, .9, .4, .6])
    aliases = pd.Series(["B", "A"])

    mean = pool(scores, images, aliases, "mean_logit")
    soft = pool(scores, images, aliases, "softmax2_logit")
    maximum = pool(scores, images, aliases, "max_logit")

    assert np.all(mean < soft)
    assert np.all(soft < maximum)
    assert np.isclose(maximum[0], .6)
    assert np.isclose(maximum[1], .9)
