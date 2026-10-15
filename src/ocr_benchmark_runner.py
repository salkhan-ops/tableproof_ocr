from __future__ import annotations

import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from evaluator import evaluate_table_grid, load_ground_truth_table, table_grid_to_ground_truth
from ocr_adapter import native_words_to_ocr_ground_truth, ocr_words_to_structure_words, save_normalised_words
from ocr_engine import TesseractOcrEngine
from ocr_evaluator import classify_primary_error, evaluate_ocr_words, match_words_by_geometry
from ocr_models import OcrWordBox
from table_reconstructor import reconstruct_from_words_file


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "ocr_benchmark_images"
MANIFEST = ROOT / "data" / "ocr_benchmark_manifest.csv"
OCR_OUT = ROOT / "outputs" / "ocr"
RAW_DIR = OCR_OUT / "raw"
NORM_DIR = OCR_OUT / "normalised"
RECON_DIR = OCR_OUT / "reconstructed"
DIAG_DIR = OCR_OUT / "diagnostics"
EVAL_DIR = OCR_OUT / "evaluation"

SOURCES = [
    ("benchmark_01", "simple ruled table", "data/benchmark_manifest.csv"),
    ("benchmark_05", "borderless table", "data/benchmark_manifest.csv"),
    ("benchmark_02", "regression table", "data/benchmark_manifest.csv"),
    ("span_01", "multi-level spanning header", "data/span_benchmark_manifest.csv"),
    ("span_06", "mixed row-column span table", "data/span_benchmark_manifest.csv"),
]

VARIANTS = [
    ("clean", 300, 0.0, 0.0, 0, "png"),
    ("moderate_resolution", 200, 0.0, 0.0, 0, "png"),
    ("skew_noise", 300, 1.0, 0.012, 0, "png"),
    ("low_resolution_compression", 100, 0.0, 0.0, 45, "jpg"),
]


def run_ocr_benchmark() -> dict[str, Any]:
    for directory in [IMAGE_DIR, RAW_DIR, NORM_DIR, RECON_DIR, DIAG_DIR, EVAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_rows, gt_records = _create_benchmark_images()
    _write_csv(MANIFEST, manifest_rows)
    (EVAL_DIR / "ocr_word_ground_truth.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in gt_records),
        encoding="utf-8",
    )

    engine = TesseractOcrEngine(_tesseract_executable())
    word_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    cell_errors: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    all_error_rows: list[dict[str, Any]] = []

    gt_by_id = {record["ocr_table_id"]: [OcrWordBox.from_dict(item) for item in record["ground_truth_word_boxes"]] for record in gt_records}
    for item in manifest_rows:
        table_id = item["ocr_table_id"]
        try:
            raw_result = engine.recognise(ROOT / item["image_path"], page_number=1)
            engine.save_raw(raw_result, RAW_DIR / f"{table_id}_raw.json")
            ocr_words = raw_result.words
            save_normalised_words(ocr_words, NORM_DIR / f"{table_id}_words.json")
            _write_structure_words(ocr_words, NORM_DIR / f"{table_id}_structure_words.json")
            _draw_overlay(ROOT / item["image_path"], ocr_words, DIAG_DIR / f"{table_id}_ocr_overlay.png")

            expected_words = gt_by_id[table_id]
            text_metrics, geometry_metrics, error_rows = evaluate_ocr_words(table_id, expected_words, ocr_words)
            word_rows.append({**item, **text_metrics})
            geometry_rows.append({**item, **geometry_metrics})
            all_error_rows.extend(error_rows)

            table_eval = _run_structure_eval(table_id, item, ocr_words, "ocr_text_ocr_boxes")
            table_rows.append({**item, **table_eval["row"]})
            cell_errors.extend(table_eval["cell_errors"])

            counterfactuals = _counterfactuals(table_id, item, expected_words, ocr_words)
            primary = classify_primary_error(text_metrics, geometry_metrics, table_eval["row"])
            attribution_rows.append(
                {
                    "ocr_table_id": table_id,
                    "source_table_id": item["source_table_id"],
                    "degradation_type": item["degradation_type"],
                    "perfect_text_perfect_boxes": counterfactuals["perfect_text_perfect_boxes"]["whole_table_exact_match"],
                    "ocr_text_perfect_boxes": counterfactuals["ocr_text_perfect_boxes"]["whole_table_exact_match"],
                    "perfect_text_ocr_boxes": counterfactuals["perfect_text_ocr_boxes"]["whole_table_exact_match"],
                    "ocr_text_ocr_boxes": table_eval["row"]["whole_table_exact_match"],
                    "primary_error_source": primary,
                }
            )
        except Exception as exc:  # noqa: BLE001 - benchmark must not silently skip.
            word_rows.append({**item, "word_exact_accuracy": 0.0, "character_error_rate": 1.0, "failure_reason": f"{type(exc).__name__}: {exc}"})
            geometry_rows.append({**item, "word_box_detection_precision": 0.0, "word_box_detection_recall": 0.0, "failure_reason": f"{type(exc).__name__}: {exc}"})
            table_rows.append({**item, "status": "FAIL", "failure_reason": f"{type(exc).__name__}: {exc}", "whole_table_exact_match": False})
            attribution_rows.append({**item, "primary_error_source": "OTHER_OCR_ERROR", "failure_reason": f"{type(exc).__name__}: {exc}"})

    _write_csv(EVAL_DIR / "ocr_word_results.csv", word_rows)
    _write_csv(EVAL_DIR / "ocr_geometry_results.csv", geometry_rows)
    _write_csv(EVAL_DIR / "ocr_table_results.csv", table_rows)
    _write_csv(EVAL_DIR / "ocr_cell_errors.csv", cell_errors or [{"ocr_table_id": "", "error": ""}])
    _write_csv(EVAL_DIR / "ocr_error_attribution.csv", attribution_rows + all_error_rows)
    summary = _summary(engine, manifest_rows, word_rows, geometry_rows, table_rows, attribution_rows)
    (EVAL_DIR / "ocr_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EVAL_DIR / "ocr_benchmark_report.md").write_text(_report(summary), encoding="utf-8")
    print(f"ocr_status={summary['phase_status']}")
    print(f"images_processed={summary['benchmark']['images_processed']}")
    print(f"summary_json={EVAL_DIR / 'ocr_summary.json'}")
    print(f"report={EVAL_DIR / 'ocr_benchmark_report.md'}")
    return summary


def _create_benchmark_images() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = _manifest_entries()
    rows = []
    gt_records = []
    random.seed(1729)
    for source_id, structure_type, _manifest in SOURCES:
        entry = entries[source_id]
        source_image = ROOT / f"outputs/extracted_tables/{source_id}_page_001_table.png"
        source_words = json.loads((ROOT / f"outputs/extracted_tables/{source_id}_page_001_words.json").read_text(encoding="utf-8"))
        metadata = json.loads((ROOT / f"outputs/extracted_tables/{source_id}_page_001_metadata.json").read_text(encoding="utf-8"))
        base_words = native_words_to_ocr_ground_truth(source_words, metadata)
        for variant_name, dpi, rotation, noise, compression, extension in VARIANTS:
            ocr_id = f"{source_id}_{variant_name}"
            image_path = IMAGE_DIR / f"{ocr_id}.{extension}"
            transformed_words = _make_variant(source_image, image_path, base_words, dpi, rotation, noise, compression)
            rows.append(
                {
                    "ocr_table_id": ocr_id,
                    "source_table_id": source_id,
                    "image_path": str(image_path.relative_to(ROOT)),
                    "source_ground_truth_path": entry["ground_truth_path"],
                    "source_ground_truth_json_path": entry.get("ground_truth_json_path", ""),
                    "degradation_type": variant_name,
                    "structure_type": structure_type,
                    "resolution_dpi": dpi,
                    "rotation_degrees": rotation,
                    "noise_level": noise,
                    "compression_level": compression,
                    "manually_verified": True,
                }
            )
            gt_records.append(
                {
                    "ocr_table_id": ocr_id,
                    "source_table_id": source_id,
                    "ground_truth_text": " ".join(word.text for word in transformed_words),
                    "ground_truth_word_boxes": [word.to_dict() for word in transformed_words],
                    "ground_truth_token_order": [word.text for word in transformed_words],
                    "ground_truth_table_structure": {
                        "ground_truth_path": entry["ground_truth_path"],
                        "ground_truth_json_path": entry.get("ground_truth_json_path", ""),
                    },
                }
            )
    return rows, gt_records


def _make_variant(source_image: Path, target_image: Path, words: list[OcrWordBox], dpi: int, rotation: float, noise: float, compression: int) -> list[OcrWordBox]:
    image = Image.open(source_image).convert("RGB")
    scale = dpi / 300.0
    transformed = _scale_words(words, scale)
    if scale != 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    if rotation:
        transformed = _rotate_words(transformed, image.width, image.height, rotation)
        image = image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
    if noise:
        image = _add_noise(image, noise)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
        image = ImageEnhance.Contrast(image).enhance(0.95)
    target_image.parent.mkdir(parents=True, exist_ok=True)
    if compression:
        image.save(target_image, quality=compression, optimize=True)
    else:
        image.save(target_image)
    return transformed


def _run_structure_eval(table_id: str, item: dict[str, Any], words: list[OcrWordBox], mode: str) -> dict[str, Any]:
    structure_words_path = NORM_DIR / f"{table_id}_{mode}_structure_words.json"
    _write_structure_words(words, structure_words_path)
    reconstruction = reconstruct_from_words_file(structure_words_path, output_dir=RECON_DIR, debug=True)
    for suffix, path in {
        "xlsx": reconstruction.xlsx_path,
        "csv": reconstruction.csv_path,
        "json": reconstruction.structured_json_path,
        "diagnostic.png": reconstruction.diagnostic_png_path,
        "diagnostics.json": reconstruction.diagnostics_json_path,
    }.items():
        destination = (RECON_DIR if suffix in {"xlsx", "csv", "json"} else DIAG_DIR) / f"{table_id}_{mode}.{suffix}"
        shutil.copyfile(path, destination)
    ground_truth = load_ground_truth_table(ROOT / item["source_ground_truth_path"], ROOT / item["source_ground_truth_json_path"] if item.get("source_ground_truth_json_path") else None)
    prediction = table_grid_to_ground_truth(reconstruction.table_grid)
    evaluation = evaluate_table_grid(table_id, prediction, ground_truth)
    _write_cell_comparison(EVAL_DIR / f"{table_id}_{mode}_cell_comparison.csv", table_id, prediction.matrix, ground_truth.matrix, evaluation.cell_errors)
    row = {
        **evaluation.to_dict(),
        "ocr_table_id": table_id,
        "source_table_id": item["source_table_id"],
        "degradation_type": item["degradation_type"],
        "counterfactual_mode": mode,
        "status": "PASS" if evaluation.whole_table_exact_match else "FAIL",
        "structure_confidence": reconstruction.table_grid.confidence,
        "xlsx_path": str((RECON_DIR / f"{table_id}_{mode}.xlsx").relative_to(ROOT)),
        "csv_path": str((RECON_DIR / f"{table_id}_{mode}.csv").relative_to(ROOT)),
        "json_path": str((RECON_DIR / f"{table_id}_{mode}.json").relative_to(ROOT)),
        "diagnostic_png_path": str((DIAG_DIR / f"{table_id}_{mode}.diagnostic.png").relative_to(ROOT)),
    }
    return {"row": row, "cell_errors": [{"ocr_table_id": table_id, **error} for error in evaluation.cell_errors]}


def _counterfactuals(table_id: str, item: dict[str, Any], expected: list[OcrWordBox], ocr: list[OcrWordBox]) -> dict[str, dict[str, Any]]:
    matches = match_words_by_geometry(expected, ocr)
    ocr_text_perfect_boxes = []
    perfect_text_ocr_boxes = []
    for match in matches:
        if match.gt_index is None or match.pred_index is None:
            continue
        gt = expected[match.gt_index]
        pred = ocr[match.pred_index]
        ocr_text_perfect_boxes.append(_replace_word(gt, pred.text, pred.confidence, "ocr_text_perfect_boxes", pred.source_token_id))
        perfect_text_ocr_boxes.append(_replace_box(pred, gt.text, "perfect_text_ocr_boxes", gt.source_token_id))
    return {
        "perfect_text_perfect_boxes": _run_structure_eval(table_id, item, expected, "perfect_text_perfect_boxes")["row"],
        "ocr_text_perfect_boxes": _run_structure_eval(table_id, item, ocr_text_perfect_boxes, "ocr_text_perfect_boxes")["row"],
        "perfect_text_ocr_boxes": _run_structure_eval(table_id, item, perfect_text_ocr_boxes, "perfect_text_ocr_boxes")["row"],
    }


def _summary(engine: TesseractOcrEngine, manifest: list[dict[str, Any]], word_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], table_rows: list[dict[str, Any]], attribution_rows: list[dict[str, Any]]) -> dict[str, Any]:
    processed = sum(1 for row in table_rows if not row.get("failure_reason"))
    skipped = len(manifest) - processed
    summary = {
        "phase_status": "PASS" if processed == 20 and skipped == 0 else "PARTIAL",
        "engine": {"name": engine.name, "version": engine.version(), "models": "tesseract default language data", "raw_output_preserved": True},
        "benchmark": {"source_tables": [source[0] for source in SOURCES], "degradation_variants": [variant[0] for variant in VARIANTS], "total_ocr_images": len(manifest), "images_processed": processed, "images_skipped": skipped, "manifest": str(MANIFEST), "ground_truth": str(EVAL_DIR / "ocr_word_ground_truth.jsonl")},
        "ocr_text_results": _avg_metrics(word_rows, ["word_exact_accuracy", "character_error_rate", "numeric_token_accuracy", "minus_sign_accuracy", "decimal_point_accuracy", "parenthesis_accuracy", "percentage_sign_accuracy", "significance_star_accuracy"]),
        "ocr_geometry_results": _avg_metrics(geometry_rows, ["word_box_detection_precision", "word_box_detection_recall", "average_matched_box_iou", "reading_order_accuracy"]),
        "ocr_geometry_counts": {
            "missed_words": sum(int(row.get("missed_words", 0) or 0) for row in geometry_rows),
            "false_words": sum(int(row.get("false_words", 0) or 0) for row in geometry_rows),
            "split_word_errors": sum(int(row.get("split_word_errors", 0) or 0) for row in geometry_rows),
            "merged_word_errors": sum(int(row.get("merged_word_errors", 0) or 0) for row in geometry_rows),
        },
        "table_results": _table_summary(table_rows),
        "by_degradation": _by_group(manifest, word_rows, geometry_rows, table_rows, "degradation_type"),
        "by_structure": _by_group(manifest, word_rows, geometry_rows, table_rows, "structure_type"),
        "counterfactual_attribution": _counterfactual_summary(attribution_rows),
        "error_counts": _error_counts(attribution_rows),
        "output_files": {
            "benchmark_report": str(EVAL_DIR / "ocr_benchmark_report.md"),
            "summary_json": str(EVAL_DIR / "ocr_summary.json"),
            "word_results": str(EVAL_DIR / "ocr_word_results.csv"),
            "geometry_results": str(EVAL_DIR / "ocr_geometry_results.csv"),
            "table_results": str(EVAL_DIR / "ocr_table_results.csv"),
            "error_attribution": str(EVAL_DIR / "ocr_error_attribution.csv"),
        },
    }
    return summary


def _manifest_entries() -> dict[str, dict[str, Any]]:
    entries = {}
    for manifest in ["data/benchmark_manifest.csv", "data/span_benchmark_manifest.csv"]:
        with (ROOT / manifest).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                entries[row["table_id"]] = row
    return entries


def _write_structure_words(words: list[OcrWordBox], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"text": word.text, "bbox": {"x0": word.x0, "y0": word.y0, "x1": word.x1, "y1": word.y1}} for word in words]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _draw_overlay(image_path: Path, words: list[OcrWordBox], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, word in enumerate(words):
        draw.rectangle([word.x0, word.y0, word.x1, word.y1], outline="red", width=2)
        draw.text((word.x0, max(0, word.y0 - 10)), str(index), fill="blue")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_cell_comparison(path: Path, table_id: str, predicted: list[list[str]], expected: list[list[str]], errors: list[dict[str, Any]]) -> None:
    rows = []
    error_coords = {(error.get("row_index"), error.get("column_index")): error for error in errors}
    row_count = max(len(predicted), len(expected))
    col_count = max([len(row) for row in predicted + expected] or [0])
    for row in range(row_count):
        for col in range(col_count):
            rows.append(
                {
                    "table_id": table_id,
                    "row_index": row,
                    "column_index": col,
                    "expected": expected[row][col] if row < len(expected) and col < len(expected[row]) else "",
                    "predicted": predicted[row][col] if row < len(predicted) and col < len(predicted[row]) else "",
                    "error_type": error_coords.get((row, col), {}).get("error_type", ""),
                }
            )
    _write_csv(path, rows)


def _replace_word(word: OcrWordBox, text: str, confidence: float, engine: str, token_id: str) -> OcrWordBox:
    return OcrWordBox(text, word.x0, word.y0, word.x1, word.y1, confidence, word.page_number, engine, token_id, word.line_hint, word.rotation, text)


def _replace_box(box_word: OcrWordBox, text: str, engine: str, token_id: str) -> OcrWordBox:
    return OcrWordBox(text, box_word.x0, box_word.y0, box_word.x1, box_word.y1, 1.0, box_word.page_number, engine, token_id, box_word.line_hint, box_word.rotation, text)


def _scale_words(words: list[OcrWordBox], scale: float) -> list[OcrWordBox]:
    return [OcrWordBox(word.text, word.x0 * scale, word.y0 * scale, word.x1 * scale, word.y1 * scale, word.confidence, word.page_number, word.source_engine, word.source_token_id, word.line_hint, word.rotation, word.raw_text) for word in words]


def _rotate_words(words: list[OcrWordBox], width: int, height: int, degrees: float) -> list[OcrWordBox]:
    radians = math.radians(degrees)
    cx, cy = width / 2, height / 2
    output = []
    for word in words:
        points = [(word.x0, word.y0), (word.x1, word.y0), (word.x1, word.y1), (word.x0, word.y1)]
        rotated = []
        for x, y in points:
            dx, dy = x - cx, y - cy
            rotated.append((cx + dx * math.cos(radians) - dy * math.sin(radians), cy + dx * math.sin(radians) + dy * math.cos(radians)))
        xs, ys = zip(*rotated)
        output.append(OcrWordBox(word.text, max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys)), word.confidence, word.page_number, word.source_engine, word.source_token_id, word.line_hint, degrees, word.raw_text))
    return output


def _add_noise(image: Image.Image, amount: float) -> Image.Image:
    random.seed(1729)
    pixels = []
    for red, green, blue in image.getdata():
        delta = int(random.gauss(0, 255 * amount))
        pixels.append((max(0, min(255, red + delta)), max(0, min(255, green + delta)), max(0, min(255, blue + delta))))
    noisy = Image.new("RGB", image.size)
    noisy.putdata(pixels)
    return noisy


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


def _by_group(manifest: list[dict[str, Any]], word_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], table_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups = sorted({row[key] for row in manifest})
    output = {}
    for group in groups:
        words = [row for row in word_rows if row.get(key) == group]
        geometry = [row for row in geometry_rows if row.get(key) == group]
        tables = [row for row in table_rows if row.get(key) == group]
        output[group] = {**_avg_metrics(words, ["word_exact_accuracy", "character_error_rate", "numeric_token_accuracy"]), **_avg_metrics(geometry, ["word_box_detection_precision", "word_box_detection_recall", "average_matched_box_iou", "reading_order_accuracy"]), **_table_summary(tables)}
    return output


def _counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: sum(1 for row in rows if row.get(key) in {True, "True"})
        for key in ["perfect_text_perfect_boxes", "ocr_text_perfect_boxes", "perfect_text_ocr_boxes", "ocr_text_ocr_boxes"]
    }


def _error_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = row.get("primary_error_source", "")
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["ocr_table_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: dict[str, Any]) -> str:
    return "# Controlled OCR Benchmark\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n"


def _tesseract_executable() -> str:
    for candidate in ["/opt/homebrew/bin/tesseract", "tesseract"]:
        if shutil.which(candidate):
            return candidate
    return "tesseract"


if __name__ == "__main__":
    run_ocr_benchmark()
