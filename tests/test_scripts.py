"""Documented entry points work from a clean, uninstalled checkout."""

import subprocess
import sys
from pathlib import Path


def test_benchmark_runs_as_documented():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/bench.py"], cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "luck check: 30 seeds" in result.stdout
    assert "death_spiral" in result.stdout
