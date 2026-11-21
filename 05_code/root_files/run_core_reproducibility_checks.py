from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_REPORT = ROOT / "CYCH_core_reproducibility_tests_20260915.json"
MARKDOWN_REPORT = ROOT / "CYCH_core_reproducibility_tests_20260915.md"

ARTIFACTS = (
    "CYCH_selected_pipeline_ablation_20260914/development_predictions_deidentified.csv",
    "CYCH_selected_pipeline_ablation_20260914/development_metrics_per_draw.csv",
    "CYCH_selected_pipeline_ablation_20260914/development_summary.csv",
    "CYCH_selected_pipeline_ablation_20260914/protocol_lock.json",
    "CYCH_reliable_negative_graph_mil_development_20260914/development_predictions.csv",
    "CYCH_reliable_negative_graph_mil_development_20260914/development_summary.csv",
    "CYCH_reliable_negative_graph_mil_development_20260914/candidate_lock.json",
    "CYCH_locked_selected_exploratory_test_20260915/exploratory_test_summary.csv",
    "CYCH_locked_selected_exploratory_test_20260915/evaluation_lock.json",
    "CYCH_locked_selected_exploratory_test_20260915/unique_patient_sensitivity_summary.csv",
    "CYCH_locked_selected_exploratory_test_20260915/unique_patient_sensitivity_bootstrap_summary.csv",
    "test_reliable_negative_graph_mil_development.py",
    "test_selected_pipeline_ablation.py",
    "test_locked_unique_patient_sensitivity.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_check_group(name: str, arguments: list[str], test_count: int, markers: list[str]) -> dict:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = time.perf_counter() - started
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    markers_found = {marker: marker in output for marker in markers}
    passed = completed.returncode == 0 and all(markers_found.values())
    return {
        "name": name,
        "test_count": test_count,
        "status": "PASS" if passed else "FAIL",
        "return_code": completed.returncode,
        "duration_seconds": round(duration, 3),
        "command": ["python", *arguments],
        "required_output_markers": markers_found,
        "output": output,
    }


def make_markdown(report: dict) -> str:
    lines = [
        "# Core reproducibility test report",
        "",
        "This report records an executable verification of the claim-bearing development ablation, reliable-negative instance loss, locked candidate reconstruction, and equal-patient sensitivity analyses. It does not constitute external validation.",
        "",
        f"- Overall status: **{report['status']}**",
        f"- Checks passed: **{report['tests_passed']}/{report['tests_total']}**",
        f"- Run time (UTC): `{report['generated_at_utc']}`",
        f"- Python: `{report['runtime']['python_version']}`",
        f"- Packages: PyTorch `{report['runtime']['packages']['torch']}`, NumPy `{report['runtime']['packages']['numpy']}`, pandas `{report['runtime']['packages']['pandas']}`, scikit-learn `{report['runtime']['packages']['scikit-learn']}`",
        "",
        "## Executed checks",
        "",
        "| Check group | Tests | Status | Time (s) |",
        "|---|---:|---|---:|",
    ]
    for group in report["groups"]:
        lines.append(
            f"| {group['name']} | {group['test_count']} | {group['status']} | {group['duration_seconds']:.3f} |"
        )

    lines.extend(
        [
            "",
            "The five standard unit tests verify negative-bag instance supervision, differentiable behavior for all-positive batches, fixed development-role membership, exact reconstruction of the locked final candidate, and reproduction of the reported development metrics. The sixth check independently recomputes the 87-patient equal-weight sensitivity estimands and fixed-pair differences from the archived locked predictions.",
            "",
            "## Authoritative artifact hashes",
            "",
            "| Project-relative file | SHA-256 |",
            "|---|---|",
        ]
    )
    for artifact in report["artifacts"]:
        lines.append(f"| `{artifact['path']}` | `{artifact['sha256']}` |")

    lines.extend(["", "## Captured output", ""])
    for group in report["groups"]:
        lines.extend([f"### {group['name']}", "", "```text", group["output"], "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    groups = [
        run_check_group(
            "Unit tests for graph MIL and selected-pipeline ablation",
            [
                "-m",
                "unittest",
                "-v",
                "test_reliable_negative_graph_mil_development.py",
                "test_selected_pipeline_ablation.py",
            ],
            5,
            ["Ran 5 tests", "OK"],
        ),
        run_check_group(
            "Equal-patient locked sensitivity assertion",
            [
                "-c",
                "import test_locked_unique_patient_sensitivity as t; "
                "t.test_equal_patient_locked_sensitivity(); "
                "print('PASS: test_equal_patient_locked_sensitivity')",
            ],
            1,
            ["PASS: test_equal_patient_locked_sensitivity"],
        ),
    ]
    missing = [relative for relative in ARTIFACTS if not (ROOT / relative).is_file()]
    artifacts = [
        {"path": relative, "sha256": sha256(ROOT / relative)}
        for relative in ARTIFACTS
        if relative not in missing
    ]
    tests_total = sum(group["test_count"] for group in groups)
    tests_passed = sum(
        group["test_count"] for group in groups if group["status"] == "PASS"
    )
    status = "PASS" if tests_passed == tests_total == 6 and not missing else "FAIL"
    report = {
        "schema_version": 1,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "Core claim-bearing development and locked-sensitivity verification",
        "tests_total": tests_total,
        "tests_passed": tests_passed,
        "runtime": {
            "implementation": sys.implementation.name,
            "python_version": sys.version.split()[0],
            "interpreter": Path(sys.executable).name,
            "packages": {
                "torch": package_version("torch"),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scikit-learn": package_version("scikit-learn"),
            },
        },
        "groups": groups,
        "missing_artifacts": missing,
        "artifacts": artifacts,
    }
    JSON_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(make_markdown(report), encoding="utf-8")
    print(f"{status}: {tests_passed}/{tests_total} checks")
    print(JSON_REPORT.name)
    print(MARKDOWN_REPORT.name)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
