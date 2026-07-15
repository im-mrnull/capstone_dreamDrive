"""Provenance rule enforcement: README results tables must be exactly what
scripts/make_tables.py generates from the committed metrics JSONs. A
hand-edited number fails this test (and CI)."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_readme_tables_match_artifacts():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_tables.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, (
        "README results tables drifted from outputs/*.json.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


def test_no_fabricated_numbers_resurface():
    """The retracted placeholder claims must not reappear in tracked docs."""
    banned = ["66% higher ADE", "131% higher FDE", "66% increase in ADE", "131% increase in FDE"]
    for doc in ["README.md", "TRAJECTRON_FIX_SUMMARY.md", "research/report.txt", "research/presentation.txt"]:
        p = REPO / doc
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for phrase in banned:
            assert phrase not in text, f"retracted claim {phrase!r} found in {doc}"
