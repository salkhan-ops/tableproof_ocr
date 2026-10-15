from __future__ import annotations

import re
from typing import Any

from tableproof.models import LineFeatureVector, TableRow, VisualLine


NUMERIC_RE = re.compile(r"^\(?[-+]?\d[\d,]*(\.\d+)?%?\)?(\*{1,3})?$")


def visual_lines_from_rows(rows: list[TableRow]) -> list[VisualLine]:
    lines: list[VisualLine] = []
    for row in rows:
        text = " ".join(cell.text for cell in row.cells if cell.text).strip()
        word_indices = [word_index for cell in row.cells for word_index in cell.source_word_indices]
        populated_columns = [cell.column_index for cell in row.cells if cell.text]
        lines.append(
            VisualLine(
                line_index=row.row_index,
                row_index=row.row_index,
                text=text,
                bounding_box=row.bounding_box,
                source_word_indices=word_indices,
                populated_columns=populated_columns,
            )
        )
    return lines


def line_feature_vectors(rows: list[TableRow], median_word_height: float) -> list[LineFeatureVector]:
    lines = visual_lines_from_rows(rows)
    vectors: list[LineFeatureVector] = []
    dominant_columns = max((len(row.cells) for row in rows), default=0)
    for index, line in enumerate(lines):
        previous_line = lines[index - 1] if index > 0 else None
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        values = _text_features(line.text)
        values.update(
            {
                "line_index": line.line_index,
                "populated_columns": line.populated_columns,
                "populated_column_count": len(line.populated_columns),
                "dominant_column_count": dominant_columns,
                "vertical_gap_from_previous": _gap(previous_line, line),
                "vertical_gap_to_next": _gap(line, next_line),
                "gap_from_previous_normalized": _gap(previous_line, line) / max(median_word_height, 1.0),
                "gap_to_next_normalized": _gap(line, next_line) / max(median_word_height, 1.0),
                "left_edge": line.bounding_box.x0 if line.bounding_box else None,
                "right_edge": line.bounding_box.x1 if line.bounding_box else None,
                "width": line.bounding_box.width if line.bounding_box else 0.0,
                "has_single_populated_column": len(line.populated_columns) == 1,
                "is_sparse": len(line.populated_columns) < dominant_columns,
                "is_first_column_only": line.populated_columns == [0],
                "is_final_line": index == len(lines) - 1,
            }
        )
        vectors.append(LineFeatureVector(line.line_index, values))
    return vectors


def _gap(left: VisualLine | None, right: VisualLine | None) -> float:
    if left is None or right is None or left.bounding_box is None or right.bounding_box is None:
        return 0.0
    return max(0.0, right.bounding_box.y0 - left.bounding_box.y1)


def _text_features(text: str) -> dict[str, Any]:
    tokens = [token for token in text.split() if token]
    numeric_tokens = [token for token in tokens if NUMERIC_RE.fullmatch(token)]
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    punctuation_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
    chars = max(sum(1 for char in text if not char.isspace()), 1)
    stripped = text.strip()
    return {
        "token_count": len(tokens),
        "average_token_length": sum(len(token) for token in tokens) / max(len(tokens), 1),
        "alphabetic_token_ratio": len(alpha_tokens) / max(len(tokens), 1),
        "numeric_token_ratio": len(numeric_tokens) / max(len(tokens), 1),
        "punctuation_ratio": punctuation_chars / chars,
        "terminal_punctuation": stripped.endswith((".", ";", ":")),
        "opening_parenthesis": "(" in stripped,
        "closing_parenthesis": ")" in stripped,
        "unmatched_parentheses": stripped.count("(") != stripped.count(")"),
        "trailing_hyphen": stripped.endswith("-"),
        "leading_lowercase": bool(stripped[:1].islower()),
        "leading_parenthetical": stripped.startswith("("),
        "significance_star_pattern": bool(re.fullmatch(r"\*{1,3}", stripped)),
        "percentage_pattern": "%" in stripped,
        "currency_pattern": bool(re.search(r"[$€£]", stripped)),
        "date_pattern": bool(re.search(r"\d{1,4}[-/]\d{1,2}", stripped)),
        "equation_like_pattern": bool(re.search(r"[=<>]", stripped)),
        "footnote_marker_pattern": bool(re.search(r"[\*†‡]|\b[a-z]\b|\d+$", stripped)),
        "title_case_pattern": stripped.istitle(),
        "all_caps_pattern": stripped.isupper() and any(char.isalpha() for char in stripped),
        "bullet_or_enum_prefix": bool(re.match(r"^(\d+\.|[a-z]\)|[-•])", stripped)),
        "note_source_prefix": stripped.lower().startswith(("note:", "notes:", "source:", "sources:")),
        "sentence_like_prose": len(tokens) >= 5 and stripped.endswith((".", ";")),
        "short_fragment": 0 < len(tokens) <= 4 and not numeric_tokens,
        "only_symbols": bool(stripped) and all(not char.isalnum() for char in stripped),
        "section_label_shape": stripped.lower().startswith(("panel ", "section ")) or stripped.endswith(":"),
    }
