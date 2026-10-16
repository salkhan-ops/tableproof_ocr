# Changelog

## v0.2.2 (2026-10-16)

### Fixed
- `render_page()` / `api.render()` serialise all PDFium calls behind a process-wide lock.
  PDFium is not thread-safe; concurrent renders from a web server's thread pool deadlocked the
  process.

## v0.2.1 (2026-10-15)

### Fixed
- OCR pages skewed by less than 1 degree are no longer resampled, but their **word
  coordinates are straightened** before reconstruction. At 0.8 degrees a label and its
  amounts drifted onto separate rows (about 5.6 pt over a 400 pt line), producing blank-label
  rows. `Diagnostics.deskew_degrees` reports the angle either way.

## v0.2.0 (2026-10-15)

Library packaging, licensing and correctness fixes found by an external evaluation on
financial-statement-style tables.

### Breaking
- Installable package `tableproof` (src layout). Library modules are imported as
  `tableproof.<module>`; the CLI, benchmark runners and audits moved to `tools/`.
- **PyMuPDF (AGPL-3.0) is no longer a runtime dependency.** Native words come from pdfplumber
  (MIT), rendering from pypdfium2 (Apache-2.0/BSD-3). PyMuPDF remains a dev-only tool that
  generates the committed fixture PDFs. Python 3.12+; Pillow 11+.

### Fixed
- **Spans no longer hide words.** A span that would cover a cell with its own text (e.g. a
  line-item label under a section heading) is rejected (`SPAN_WOULD_HIDE_WORDS`).
  `no_word_loss` is now computed on the final grid; any word missing after span resolution
  fails it (`WORDS_HIDDEN_BY_SPAN`). Previously such labels vanished while it reported a pass.
- **OCR TSV parsing.** Tesseract output is read with `csv.QUOTE_NONE`. A recognised `"` used to
  open a CSV quoted field that swallowed the rest of the page (91 of 110 words lost on a clean
  statement scan).
- OCR errors no longer include paths or Tesseract's stderr; the raw TSV (document content) is
  kept only when asked (`keep_raw_response=True`).
- The OCR enhancement runner's `phase_status` was a hard-coded `PASS`; it is now computed.

### Ground truth corrected
- `span_09`: its spec covered "Cash", "Inventory", "Property" and "Payables" with vertical
  "Current assets"-style spans, so the frozen v0.1.0 baseline certified that those labels
  disappear. The column-1 subsection headings are rows, not spans; the ground truth and
  manifest were regenerated from the corrected spec. The structure regression is 29/29
  against it.

### Added
- `tableproof.api`: `native_page()`, `ocr_page()`, `render()`. Pure, in-memory, frozen typed
  results (`TableResult`, `Cell`, `Diagnostics`); boxes in PDF points; `raw_text` and `text`
  per cell; per-cell confidence (lowest OCR word confidence).
- `ReconstructionOptions(profile="financial_statement")`: unresolved lines (section headings)
  stay in the grid as flagged rows, so every label reaches the caller.
- Significance-marker correction is opt-in in the API (`correct_markers=False` by default).
- Conservative deskew (projection profiles; only at 1 degree or more).
- `TesseractOcrEngine(languages=..., tessdata_dir=...)`, `recognise_bytes()` (stdin).
- Financial-statement fixtures and gates: no hidden words, every label in the grid, amounts
  verbatim; OCR gate on rendered and degraded scans (clean 98.0%, hard 91.7% of amounts read
  verbatim; gates 95% / 88%).
- `NOTICE`, `CHANGELOG.md`, `py.typed`, `__version__`.

## v0.1.0-born-digital-structure-baseline (2026-07-25)
Frozen born-digital structure baseline, imported into git as found. See `STRUCTURE_BASELINE.md`.
