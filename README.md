# TableProof OCR

Table extraction from annual accounts, reliably.

Baseline extraction and deterministic native-text table reconstruction tools for born-digital PDFs.

This project intentionally stops before OCR, ML table recognition, semantic validation, arithmetic checking, or automatic correction. The extraction CLI takes a PDF, a page number, and an optional table bounding box. It preserves:

- native PDF words with coordinates as JSON
- a high-resolution PNG rendering of the selected table area
- page metadata as JSON

The reconstruction CLI consumes the native words JSON and builds a deterministic table-grid baseline from word bounding boxes.

## Project Structure

```text
data/raw_pdfs/
data/table_images/
data/ground_truth/
outputs/extracted_tables/
outputs/diagnostics/
outputs/evaluation/
src/tableproof/   # the library (installable package)
tools/            # CLI, benchmark runners and audits (write to outputs/)
tests/
```

## Setup

```bash
uv sync            # Python 3.12+; installs tableproof (editable) and the dev tools
uv run pytest
```

Library use: `pip install tableproof` from a release wheel, then `import tableproof`.

## Extraction CLI

Extract the full page:

```bash
python tools/cli.py data/raw_pdfs/sample.pdf 1
```

Extract a table area:

```bash
python tools/cli.py data/raw_pdfs/sample.pdf 1 --bbox 72,120,520,360
```

Bounding boxes use PDF point coordinates in `x0,y0,x1,y1` order. Page numbers are 1-based.

By default, outputs are written to `outputs/extracted_tables/`:

- `{pdf_stem}_page_001_table.png`
- `{pdf_stem}_page_001_words.json`
- `{pdf_stem}_page_001_metadata.json`

## Reconstruction CLI

Reconstruct a table grid from existing native words JSON:

```bash
python tools/cli.py reconstruct outputs/extracted_tables/sample_page_001_words.json
```

Optional overrides:

```bash
python tools/cli.py reconstruct outputs/extracted_tables/sample_page_001_words.json \
  --output-dir outputs/evaluation \
  --row-tolerance 8 \
  --column-tolerance 12 \
  --debug
```

Default thresholds are adaptive. Manual tolerances override only the relevant adaptive threshold.

The reconstructor writes:

- `{words_stem}_reconstructed.xlsx`
- `{words_stem}_reconstructed.csv`
- `{words_stem}_reconstructed.json`
- `{words_stem}_diagnostic.png`
- `{words_stem}_diagnostics.json`

The XLSX workbook preserves values as strings, applies basic borders, and includes `Diagnostics` and `Notes` worksheets. The reconstructor also records raw visual lines, continuation candidates, selected continuation decisions, unresolved lines, and no-word-loss diagnostics in JSON.

Continuation handling is deterministic and native-text only. It can resolve wrapped cells, multi-line headers, hyphenated line breaks, and multiple wrapped cells on one visual line. Ambiguous trailing lines are preserved and flagged instead of silently merged.

## Benchmark

Run the five-table born-digital benchmark:

```bash
python tools/benchmark_runner.py --manifest data/benchmark_manifest.csv
```

The benchmark includes CSV ground truth for ordinary tables and XLSX plus structured JSON ground truth for merged-cell tables. It writes table-level predictions, cell comparisons, diagnostics, aggregate metrics, and a markdown report under `outputs/evaluation/` and `outputs/diagnostics/`.

The benchmark runner includes a deterministic note-row exclusion diagnostic mode. Rows beginning with note/source/footnote labels are recorded in diagnostics and excluded from benchmark scoring when they are sparse note-like rows below a table.

Merged-cell evaluation distinguishes physical rows and columns from logical cells. It measures exact merged-range accuracy, row/column span precision and recall, span-text accuracy, covered-coordinate accuracy, and span placement accuracy.

Run the continuation benchmark fixtures:

```bash
python scripts/create_continuation_benchmark_fixtures.py
python tools/continuation_benchmark_runner.py
```

This writes continuation table results, decision comparisons, errors, a JSON summary, and a markdown report under `outputs/evaluation/`.

## OCR (post-baseline, unfrozen)

Everything above is the frozen `v0.1.0-born-digital-structure-baseline` (see `STRUCTURE_BASELINE.md`), which is native-text only and explicitly excludes OCR. A separate OCR layer was added after that freeze and is not yet part of a versioned baseline.

The OCR layer runs Tesseract (`src/tableproof/ocr_engine.py`) over rendered table images, with deterministic image preprocessing (`src/tableproof/ocr_image_preprocessor.py`) and a significance-marker correction pass (`tools/ocr_enhancement_runner.py`) on top of the raw OCR output.

Run the OCR benchmark (renders 5 source tables x 4 degradation variants = 20 images from `data/ocr_benchmark_manifest.csv`, runs raw Tesseract, and scores against ground truth):

```bash
python tools/ocr_benchmark_runner.py
```

Run the enhanced pass (preprocessing + significance-marker correction) and compare against the raw baseline:

```bash
python tools/ocr_enhancement_runner.py
```

Both write word/table/geometry results, an error-attribution CSV, a JSON summary, and a markdown report under `outputs/ocr/evaluation/` and `outputs/ocr_enhanced/evaluation/`.

As of the last run, the enhanced pass reports 98.75% exact cell-text accuracy, 100% numeric/empty-cell/merged-range accuracy, and 16/20 whole-table exact matches (up from 11/20 raw), with `phase_status: PASS` in `outputs/ocr_enhanced/evaluation/ocr_enhanced_summary.json`.

**Coverage caveat:** the OCR benchmark only exercises 5 of the born-digital baseline's source tables (simple ruled, borderless, regression, multi-level header, mixed row-column span) under mild synthetic degradation (rotation up to 1 degree, noise up to 0.012, resolution down to 100 DPI, light JPEG compression). It does not cover footnote/source-note exclusion, multi-line cell continuations, staggered/asymmetric headers, hierarchical stub groups, heavier skew or noise, or real scanned documents. Treat OCR results as promising on easy cases, not as a validated general baseline.

## Test

```bash
pytest
```

The tests create synthetic native-word tables and small born-digital PDFs. They verify extraction artifacts, row and column counts, cell text, empty-cell placement, deterministic reconstruction, footnote separation, merged spans, continuation joining modes, unresolved-line diagnostics, no-word-loss accounting, and benchmark regressions.

## License

Apache License 2.0. See `LICENSE`.
