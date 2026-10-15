from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from continuation_benchmark_runner import run_continuation_benchmark
from scripts.create_continuation_benchmark_fixtures import main as create_continuation_fixtures


def test_continuation_benchmark_regression_suite_passes() -> None:
    create_continuation_fixtures()

    result = run_continuation_benchmark()
    summary = result["summary"]

    assert summary["tables_total"] == 14
    assert summary["tables_passed"] == 14
    assert summary["false_merges"] == 0
    assert summary["missed_merges"] == 0
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["final_merged_text_accuracy"] == 1.0
    assert summary["numeric_cell_accuracy"] == 1.0
