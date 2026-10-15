from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from evaluator import evaluate_table_grid, load_ground_truth_table, table_grid_to_ground_truth
from tableproof.ocr_adapter import OcrWordBox, correct_significance_markers, save_normalised_words
from ocr_benchmark_runner import (
    DIAG_DIR as BASE_DIAG_DIR,
    EVAL_DIR as BASE_EVAL_DIR,
    MANIFEST,
    ROOT,
    _draw_overlay,
    _tesseract_executable,
    _write_cell_comparison,
    _write_structure_words,
)
from tableproof.ocr_engine import TesseractOcrEngine
from ocr_evaluator import classify_primary_error, evaluate_ocr_words, match_words_by_geometry
from tableproof.ocr_image_preprocessor import preprocess_for_tesseract
from tableproof.table_reconstructor import reconstruct_from_words_file


OUT = ROOT / "outputs" / "ocr_enhanced"
PRE_DIR = OUT / "preprocessed"
RAW_DIR = OUT / "raw"
NORM_DIR = OUT / "normalised"
RECON_DIR = OUT / "reconstructed"
DIAG_DIR = OUT / "diagnostics"
EVAL_DIR = OUT / "evaluation"


def run_ocr_enhancement() -> dict[str, Any]:
    for directory in [PRE_DIR, RAW_DIR, NORM_DIR, RECON_DIR, DIAG_DIR, EVAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv(MANIFEST)
    gt_records = [
        json.loads(line)
        for line in (BASE_EVAL_DIR / "ocr_word_ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gt_by_id = {
        record["ocr_table_id"]: [OcrWordBox.from_dict(item) for item in record["ground_truth_word_boxes"]]
        for record in gt_records
    }
    engine = TesseractOcrEngine(_tesseract_executable())

    word_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    cell_errors: list[dict[str, Any]] = []

    for item in manifest_rows:
        table_id = item["ocr_table_id"]
        source_image = ROOT / item["image_path"]
        preprocessed = preprocess_for_tesseract(source_image, PRE_DIR / f"{table_id}_preprocessed.png")
        scale_x, scale_y = _scale_factors(source_image, preprocessed)
        raw = engine.recognise(preprocessed, page_number=1)
        engine.save_raw(raw, RAW_DIR / f"{table_id}_raw.json")
        words = [_rescale_word(word, scale_x, scale_y) for word in raw.words]
        words = [correct_significance_markers(word) for word in words]
        significance_marker_corrections = sum(
            1 for before, after in zip(raw.words, words) if before.text != after.text
        )
        save_normalised_words(words, NORM_DIR / f"{table_id}_words.json")
        _write_structure_words(words, NORM_DIR / f"{table_id}_structure_words.json")
        _draw_overlay(source_image, words, DIAG_DIR / f"{table_id}_ocr_overlay.png")

        expected = gt_by_id[table_id]
        text_metrics, geometry_metrics, errors = evaluate_ocr_words(table_id, expected, words)
        word_rows.append({**item, **text_metrics, "significance_marker_corrections": significance_marker_corrections})
        geometry_rows.append({**item, **geometry_metrics})

        table_eval = _run_structure_eval(table_id, item, words)
        table_rows.append({**item, **table_eval["row"]})
        cell_errors.extend(table_eval["cell_errors"])
        attribution_rows.append(
            {
                "ocr_table_id": table_id,
                "source_table_id": item["source_table_id"],
                "degradation_type": item["degradation_type"],
                "primary_error_source": classify_primary_error(text_metrics, geometry_metrics, table_eval["row"]),
                "whole_table_exact_match": table_eval["row"]["whole_table_exact_match"],
            }
        )
        attribution_rows.extend(errors)

    _write_csv(EVAL_DIR / "ocr_enhanced_word_results.csv", word_rows)
    _write_csv(EVAL_DIR / "ocr_enhanced_geometry_results.csv", geometry_rows)
    _write_csv(EVAL_DIR / "ocr_enhanced_table_results.csv", table_rows)
    _write_csv(EVAL_DIR / "ocr_enhanced_cell_errors.csv", cell_errors or [{"ocr_table_id": "", "error": ""}])
    _write_csv(EVAL_DIR / "ocr_enhanced_error_attribution.csv", attribution_rows)
    summary = _summary(word_rows, geometry_rows, table_rows, attribution_rows)
    (EVAL_DIR / "ocr_enhanced_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EVAL_DIR / "ocr_enhanced_report.md").write_text("# OCR Enhancement Pass\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ocr_enhancement_status={summary['phase_status']}")
    print(f"images_processed={summary['images_processed']}")
    print(f"whole_table_exact_matches={summary['table_results']['whole_table_exact_matches']}")
    print(f"summary_json={EVAL_DIR / 'ocr_enhanced_summary.json'}")
    return summary


def _run_structure_eval(table_id: str, item: dict[str, Any], words: list[OcrWordBox]) -> dict[str, Any]:
    structure_words_path = NORM_DIR / f"{table_id}_structure_words.json"
    _write_structure_words(words, structure_words_path)
    reconstruction = reconstruct_from_words_file(structure_words_path, output_dir=RECON_DIR, debug=True)
    for suffix, path in {
        "xlsx": reconstruction.xlsx_path,
        "csv": reconstruction.csv_path,
        "json": reconstruction.structured_json_path,
        "diagnostic.png": reconstruction.diagnostic_png_path,
        "diagnostics.json": reconstruction.diagnostics_json_path,
    }.items():
        destination = (RECON_DIR if suffix in {"xlsx", "csv", "json"} else DIAG_DIR) / f"{table_id}.{suffix}"
        shutil.copyfile(path, destination)
    ground_truth = load_ground_truth_table(
        ROOT / item["source_ground_truth_path"],
        ROOT / item["source_ground_truth_json_path"] if item.get("source_ground_truth_json_path") else None,
    )
    prediction = table_grid_to_ground_truth(reconstruction.table_grid)
    evaluation = evaluate_table_grid(table_id, prediction, ground_truth)
    _write_cell_comparison(EVAL_DIR / f"{table_id}_cell_comparison.csv", table_id, prediction.matrix, ground_truth.matrix, evaluation.cell_errors)
    row = {
        **evaluation.to_dict(),
        "ocr_table_id": table_id,
        "source_table_id": item["source_table_id"],
        "degradation_type": item["degradation_type"],
        "status": "PASS" if evaluation.whole_table_exact_match else "FAIL",
        "structure_confidence": reconstruction.table_grid.confidence,
    }
    return {"row": row, "cell_errors": [{"ocr_table_id": table_id, **error} for error in evaluation.cell_errors]}


def _summary(word_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], table_rows: list[dict[str, Any]], attribution_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = json.loads((BASE_EVAL_DIR / "ocr_summary.json").read_text(encoding="utf-8"))
    summary = {
        "phase_status": "PASS",
        "enhancement": "deterministic_tesseract_preprocessing_and_significance_marker_correction",
        "images_processed": len(table_rows),
        "significance_marker_corrections_applied": sum(
            int(row.get("significance_marker_corrections", 0) or 0) for row in word_rows
        ),
        "images_skipped": 0,
        "baseline_summary": str(BASE_EVAL_DIR / "ocr_summary.json"),
        "word_results": _avg_metrics(word_rows, ["word_exact_accuracy", "character_error_rate", "numeric_token_accuracy", "significance_star_accuracy"]),
        "geometry_results": _avg_metrics(geometry_rows, ["word_box_detection_precision", "word_box_detection_recall", "average_matched_box_iou", "reading_order_accuracy"]),
        "table_results": _table_summary(table_rows),
        "baseline_comparison": {},
        "by_degradation": _by_group(word_rows, geometry_rows, table_rows, "degradation_type"),
        "by_structure": _by_group(word_rows, geometry_rows, table_rows, "structure_type"),
        "error_counts": _error_counts(attribution_rows),
        "output_files": {
            "summary_json": str(EVAL_DIR / "ocr_enhanced_summary.json"),
            "report": str(EVAL_DIR / "ocr_enhanced_report.md"),
            "word_results": str(EVAL_DIR / "ocr_enhanced_word_results.csv"),
            "geometry_results": str(EVAL_DIR / "ocr_enhanced_geometry_results.csv"),
            "table_results": str(EVAL_DIR / "ocr_enhanced_table_results.csv"),
            "error_attribution": str(EVAL_DIR / "ocr_enhanced_error_attribution.csv"),
            "diagnostics": str(DIAG_DIR),
        },
    }
    for key, value in summary["table_results"].items():
        if isinstance(value, (int, float)):
            summary["baseline_comparison"][key] = {
                "baseline": baseline["table_results"].get(key),
                "enhanced": value,
                "delta": round(value - baseline["table_results"].get(key, 0), 6),
            }
    return summary


def _scale_factors(source: Path, preprocessed: Path) -> tuple[float, float]:
    from PIL import Image

    source_image = Image.open(source)
    pre_image = Image.open(preprocessed)
    return pre_image.width / source_image.width, pre_image.height / source_image.height


def _rescale_word(word: OcrWordBox, scale_x: float, scale_y: float) -> OcrWordBox:
    return OcrWordBox(
        text=word.text,
        x0=word.x0 / scale_x,
        y0=word.y0 / scale_y,
        x1=word.x1 / scale_x,
        y1=word.y1 / scale_y,
        confidence=word.confidence,
        page_number=word.page_number,
        source_engine=f"{word.source_engine}_preprocessed",
        source_token_id=word.source_token_id,
        line_hint=word.line_hint,
        rotation=word.rotation,
        raw_text=word.raw_text,
    )


def _avg_metrics(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    return {key: _avg(rows, key) for key in keys}


def _table_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tables_with_exact_dimensions": sum(1 for row in rows if row.get("exact_row_count_match") in {True, "True"} and row.get("exact_column_count_match") in {True, "True"}),
        "exact_cell_text_accuracy": _avg(rows, "exact_cell_text_accuracy"),
        "numeric_cell_accuracy": _avg(rows, "numeric_cell_accuracy"),
        "placement_rate": _avg(rows, "correct_row_and_column_placement_rate"),
        "empty_cell_accuracy": _avg(rows, "empty_cell_accuracy"),
        "exact_merged_range_accuracy": _avg(rows, "exact_merged_range_accuracy"),
        "whole_table_exact_matches": sum(1 for row in rows if row.get("whole_table_exact_match") in {True, "True"}),
        "corrections_required": sum(int(float(row.get("cells_requiring_manual_correction", 0) or 0)) for row in rows),
    }


def _by_group(word_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], table_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups = sorted({row[key] for row in word_rows})
    return {
        group: {
            **_avg_metrics([row for row in word_rows if row.get(key) == group], ["word_exact_accuracy", "character_error_rate", "numeric_token_accuracy"]),
            **_avg_metrics([row for row in geometry_rows if row.get(key) == group], ["word_box_detection_precision", "word_box_detection_recall", "average_matched_box_iou"]),
            **_table_summary([row for row in table_rows if row.get(key) == group]),
        }
        for group in groups
    }


def _error_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = row.get("primary_error_source") or row.get("error_type")
        if source:
            counts[source] = counts.get(source, 0) + 1
    return counts


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return round(sum(values) / len(values), 6) if values else 0.0


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["ocr_table_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_ocr_enhancement()
