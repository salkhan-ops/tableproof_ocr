"""Characters with a rounding-level skew are horizontal text, not rotated text."""

from tableproof.pdf_extractor import straighten_chars


def _char(matrix: tuple[float, ...], upright: bool) -> dict[str, object]:
    return {"text": "A", "matrix": matrix, "upright": upright}


def test_rounding_skew_is_upright() -> None:
    chars = [_char((0.804, -9.09e-08, -3.94e-09, 0.804, 43.5, 707.8), False)]
    assert straighten_chars(chars) == 1
    assert chars[0]["upright"] is True


def test_real_rotation_and_mirroring_are_kept() -> None:
    rotated = _char((0.0, 1.0, -1.0, 0.0, 10.0, 10.0), False)
    skewed = _char((1.0, 0.05, 0.0, 1.0, 0.0, 0.0), False)
    mirrored = _char((-1.0, 0.0, 0.0, 1.0, 0.0, 0.0), False)
    assert straighten_chars([rotated, skewed, mirrored]) == 0
    assert not any(c["upright"] for c in (rotated, skewed, mirrored))


def test_upright_chars_are_untouched() -> None:
    chars = [_char((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), True)]
    assert straighten_chars(chars) == 0
