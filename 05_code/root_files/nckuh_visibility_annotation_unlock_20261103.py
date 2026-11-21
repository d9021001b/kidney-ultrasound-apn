"""Controlled unlock of completed, adjudicated NCKUH visibility annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nckuh_visibility_annotation_qc_20261103 import COMPARABLE


def unlock(consensus_path: Path, private_key_path: Path, out: Path) -> None:
    """Join only final blind consensus to the private DMSA key after QC.

    The caller must manually resolve each blank consensus field before this
    function is permitted to read the private key.
    """
    consensus = pd.read_csv(consensus_path, dtype=str).fillna("")
    expected = {"case_id", "needs_adjudication", *COMPARABLE}
    if set(consensus.columns) != expected or consensus.case_id.duplicated().any():
        raise ValueError("Consensus file has unexpected columns or duplicate case IDs")
    blanks = consensus[list(COMPARABLE)].eq("")
    if blanks.any().any():
        raise ValueError("Consensus remains incomplete; resolve every disagreement before unlocking")
    key = pd.read_csv(private_key_path, dtype=str)
    if set(key.columns) != {"case_id", "patient_alias", "DMSA_APN"} or key.case_id.duplicated().any():
        raise ValueError("Unexpected private-key schema")
    merged = key.merge(consensus, on="case_id", how="inner", validate="one_to_one")
    if len(merged) != len(key):
        raise ValueError("Consensus and private key do not have the same cases")
    out.mkdir(parents=True, exist_ok=False)
    targets = pd.concat([
        pd.DataFrame({"patient_alias": merged.patient_alias, "DMSA_APN": merged.DMSA_APN, "side": side,
                      "visible_abnormality": merged[f"{side}_visible_abnormality"].str.lower().eq("yes").astype(int),
                      "laterality_confirmed": merged[f"{side}_laterality_confirmed"], "finding_type": merged[f"{side}_finding_type"],
                      "region": merged[f"{side}_region"], "confidence_1_to_5": merged[f"{side}_confidence_1_to_5"],
                      "image_quality_exclusion": merged.image_quality_exclusion})
        for side in ("left", "right")
    ], ignore_index=True)
    targets.to_csv(out / "visibility_targets_PRIVATE.csv", index=False)
    (out / "unlock_audit.json").write_text(json.dumps({"cases": len(merged), "kidney_sides": len(targets), "all_consensus_fields_complete": True}, indent=2), encoding="utf-8")
    print(f"Unlocked {len(targets)} adjudicated kidney-side visibility targets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consensus", type=Path)
    parser.add_argument("private_key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    unlock(args.consensus, args.private_key, args.out)
