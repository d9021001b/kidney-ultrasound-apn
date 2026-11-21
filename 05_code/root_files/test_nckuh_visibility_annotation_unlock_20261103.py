import pandas as pd
import pytest

from nckuh_visibility_annotation_qc_20261103 import COMPARABLE
from nckuh_visibility_annotation_unlock_20261103 import unlock


def test_unlock_refuses_incomplete_consensus(tmp_path):
    consensus = pd.DataFrame([{ "case_id": "V001", "needs_adjudication": "False", **{field: "" for field in COMPARABLE} }])
    key = pd.DataFrame([{ "case_id": "V001", "patient_alias": "P001", "DMSA_APN": "N" }])
    consensus_path, key_path = tmp_path / "consensus.csv", tmp_path / "key.csv"
    consensus.to_csv(consensus_path, index=False); key.to_csv(key_path, index=False)
    with pytest.raises(ValueError, match="incomplete"):
        unlock(consensus_path, key_path, tmp_path / "out")


def test_unlock_writes_one_target_per_kidney_side(tmp_path):
    consensus = pd.DataFrame([{ "case_id": "V001", "needs_adjudication": "False", **{field: "yes" for field in COMPARABLE} }])
    key = pd.DataFrame([{ "case_id": "V001", "patient_alias": "P001", "DMSA_APN": "N" }])
    consensus_path, key_path, out = tmp_path / "consensus.csv", tmp_path / "key.csv", tmp_path / "out"
    consensus.to_csv(consensus_path, index=False); key.to_csv(key_path, index=False)
    unlock(consensus_path, key_path, out)
    targets = pd.read_csv(out / "visibility_targets_PRIVATE.csv")
    assert len(targets) == 2 and set(targets.side) == {"left", "right"}
