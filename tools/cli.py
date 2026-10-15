from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tableproof.models import BoundingBox
from tableproof.pdf_extractor import DEFAULT_DPI, PdfExtractionError, extract_table_area
from tableproof.table_reconstructor import ReconstructionError, reconstruct_from_words_file


def build_extraction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract native PDF words and a rendered PNG crop for one table area."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a born-digital PDF")
    parser.add_argument("page_number", type=int, help="1-based page number to extract")
    parser.add_argument(
        "--bbox",
        type=BoundingBox.from_csv,
        help="Optional table bounding box in PDF points: x0,y0,x1,y1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/extracted_tables"),
        help="Directory for PNG, words JSON, and metadata JSON outputs",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render DPI for the cropped table PNG (default: {DEFAULT_DPI})",
    )
    return parser


def build_reconstruct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct a deterministic table grid from native words JSON."
    )
    parser.add_argument("command", choices=["reconstruct"])
    parser.add_argument("words_json_path", type=Path, help="Path to native words JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory for XLSX, CSV, JSON, and diagnostics outputs",
    )
    parser.add_argument("--debug", action="store_true", help="Include debug metrics in outputs")
    parser.add_argument(
        "--row-tolerance",
        type=float,
        help="Manual row clustering tolerance in PDF points",
    )
    parser.add_argument(
        "--column-tolerance",
        type=float,
        help="Manual column alignment tolerance in PDF points",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "reconstruct":
        parser = build_reconstruct_parser()
        args = parser.parse_args(argv)
        try:
            result = reconstruct_from_words_file(
                words_json_path=args.words_json_path,
                output_dir=args.output_dir,
                debug=args.debug,
                row_tolerance=args.row_tolerance,
                column_tolerance=args.column_tolerance,
            )
        except (ReconstructionError, ValueError) as exc:
            parser.exit(status=2, message=f"error: {exc}\n")

        table_grid = result.table_grid
        print(f"xlsx={result.xlsx_path}")
        print(f"csv={result.csv_path}")
        print(f"structured_json={result.structured_json_path}")
        print(f"diagnostic_png={result.diagnostic_png_path}")
        print(f"diagnostics_json={result.diagnostics_json_path}")
        print(f"row_count={table_grid.row_count}")
        print(f"column_count={table_grid.column_count}")
        print(f"overall_confidence={table_grid.confidence}")
        if table_grid.warnings:
            print("warnings=" + ",".join(warning.code for warning in table_grid.warnings))
        else:
            print("warnings=")
        print("preview:")
        for row in table_grid.matrix():
            print(" | ".join(cell or "" for cell in row))
        return 0

    parser = build_extraction_parser()
    args = parser.parse_args(argv)

    try:
        result = extract_table_area(
            pdf_path=args.pdf_path,
            page_number=args.page_number,
            bbox=args.bbox,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
    except (PdfExtractionError, ValueError) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    print(f"cropped_table_png={result.table_png_path}")
    print(f"native_words_json={result.words_json_path}")
    print(f"page_metadata_json={result.metadata_json_path}")
    print(f"word_count={result.word_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
