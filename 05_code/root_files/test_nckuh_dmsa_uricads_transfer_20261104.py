from pathlib import Path

import numpy as np

from nckuh_dmsa_uricads_transfer_20261104 import LABEL_NAMES, parse_label


ROOT = Path(__file__).resolve().parents[2]


def test_public_label_parser_matches_known_examples():
    labels, view = parse_label(ROOT / "third_party/URI-CADS/bbdd/labels/1.txt")
    np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0, 0, 1])
    assert view == 0

    labels, view = parse_label(ROOT / "third_party/URI-CADS/bbdd/labels/51.txt")
    np.testing.assert_array_equal(labels, [1, 0, 0, 1, 0, 0, 0])
    assert view == 0
    assert len(LABEL_NAMES) == 7
