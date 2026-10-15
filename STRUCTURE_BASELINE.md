# TableProof OCR Structure Baseline

Version/tag: `v0.1.0-born-digital-structure-baseline`

Date: 2026-07-25

Git commit hash: unavailable - this workspace is not a git repository.

## Verified Results

- Pytest: 86 passed, 0 failed
- Benchmark tables: 29
- Tables passed: 29
- Whole-table exact matches: 29
- Exact cell-text accuracy: 1.00
- Numeric-cell accuracy: 1.00
- Placement rate: 1.00
- Empty-cell accuracy: 1.00
- False spans: 0
- Missed spans: 0
- Lost words: 0
- Duplicated words: 0
- Conflicting assignments: 0
- Irreversible decisions: 0

## Supported Structures

- Native born-digital word extraction with bounding boxes
- Deterministic row and column reconstruction
- Borderless tables
- Footnote and source-note exclusion
- Multi-line cells and row continuations
- Genuine empty-cell preservation
- Column spans and row spans
- Multi-level, asymmetric, and staggered headers
- Hierarchical body row groups and stub spans
- Regression panels and grouped model columns
- Mixed row-column spans

## Verified Limitations

- Born-digital structure only
- No OCR in the frozen structure baseline (a separate, unfrozen OCR layer exists post-baseline — see `README.md` § OCR)
- No cross-page reconstruction
- No rotated-text reconstruction
- No semantic validation
- No arithmetic validation
- No automatic correction

## Relevant Output Paths

- Combined report: `outputs/evaluation/complex_span_benchmark_report.md`
- Combined summary: `outputs/evaluation/complex_span_summary.json`
- Span report: `outputs/evaluation/span_benchmark_report.md`
- Span summary: `outputs/evaluation/span_summary.json`
- Baseline checksums: `outputs/baseline/v0.1.0-born-digital-structure-baseline/checksums.sha256`

## Regression Command

```bash
.venv/bin/python scripts/run_structure_regression.py
```
