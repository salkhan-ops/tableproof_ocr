from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commands = [
        [sys.executable, "-m", "pytest"],
        [sys.executable, "tools/span_audit.py"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True)
        if result.returncode != 0:
            return result.returncode
    summary = json.loads((ROOT / "outputs/evaluation/complex_span_summary.json").read_text(encoding="utf-8"))
    required = {
        "tables_total": 29,
        "tables_passed": 29,
        "tables_failed": 0,
        "whole_table_exact_matches": 29,
        "exact_cell_text_accuracy": 1.0,
        "numeric_cell_accuracy": 1.0,
        "placement_rate": 1.0,
        "empty_cell_accuracy": 1.0,
        "corrections_required": 0,
    }
    failures = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in required.items()
        if summary.get(key) != expected
    }
    span = summary.get("span_summary", {})
    for key in ["false_spans", "missed_spans", "lost_words", "duplicated_words", "conflicting_assignments", "irreversible_span_decisions"]:
        if span.get(key) != 0:
            failures[key] = {"expected": 0, "actual": span.get(key)}
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
