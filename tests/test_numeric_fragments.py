"""Amounts split by padding spaces are joined; separate columns are not."""

from tableproof.pdf_extractor import merge_numeric_fragments


def _w(text: str, x0: float, x1: float, top: float = 100.0) -> dict[str, object]:
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": top + 8.7}


def test_touching_numeric_pieces_become_one_amount() -> None:
    words = [_w("receivable", 88.1, 127.0), _w("2", 452.34, 457.22), _w(",576,613", 457.16, 489.1)]
    merged = merge_numeric_fragments(words)
    assert [w["text"] for w in merged] == ["receivable", "2,576,613"]
    assert merged[1]["x0"] == 452.34
    assert merged[1]["x1"] == 489.1


def test_amounts_in_separate_columns_stay_apart() -> None:
    words = [_w("1,385,742", 452.0, 489.0), _w("2,294,994", 537.1, 575.8)]
    assert [w["text"] for w in merge_numeric_fragments(words)] == ["1,385,742", "2,294,994"]


def test_text_and_other_lines_are_never_joined() -> None:
    words = [_w("Note", 400.0, 420.0), _w("5", 420.5, 424.0), _w("7", 425.0, 428.0, top=120.0)]
    assert [w["text"] for w in merge_numeric_fragments(words)] == ["Note", "5", "7"]


def test_out_of_order_columns_on_one_line_stay_apart() -> None:
    # Tops differ slightly, so the right-hand amount sorts first (real filing, PSX 272584).
    words = [_w("11,000,000", 509.2, 546.0, top=200.04), _w("11,000,000", 447.1, 483.9, top=200.0)]
    texts = sorted(w["text"] for w in merge_numeric_fragments(words))
    assert texts == ["11,000,000", "11,000,000"]
