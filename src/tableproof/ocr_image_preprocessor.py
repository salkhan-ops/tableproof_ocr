from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def preprocess_for_tesseract(
    image_path: Path | str,
    output_path: Path | str,
    target_min_width: int = 3000,
) -> Path:
    """Create a deterministic OCR-friendly copy without changing geometry metadata."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = Image.open(image_path).convert("L")

    scale = max(1.0, target_min_width / max(image.width, 1))
    if scale > 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    # Denoise after upscaling, not before: a median filter applied to the
    # native low-resolution/JPEG-compressed image erases thin character
    # strokes outright, while the same filter on the upscaled image only
    # smooths JPEG block artifacts. Order was empirically verified.
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.1, percent=160, threshold=3))
    image = ImageEnhance.Contrast(image).enhance(1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return output_path


def preprocess_image(image: Image.Image, target_min_width: int = 3000) -> Image.Image:
    """In-memory version of :func:`preprocess_for_tesseract` (same steps, nothing written)."""
    image = image.convert("L")
    scale = max(1.0, target_min_width / max(image.width, 1))
    if scale > 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.1, percent=160, threshold=3))
    return ImageEnhance.Contrast(image).enhance(1.2)


def estimate_skew(image: Image.Image, max_degrees: float = 3.0, step: float = 0.1) -> float:
    """Skew angle in degrees (rotate by the negative to straighten), by projection profiles.

    Text lines are sharpest (row sums vary most) when horizontal. Deterministic; estimated on
    a downscaled copy so it stays fast on full pages.
    """
    gray = image.convert("L")
    small = gray.resize((max(1, gray.width // 4), max(1, gray.height // 4)))
    ink = small.point(lambda v: 255 if v < 160 else 0)
    best_angle, best_score = 0.0, -1.0
    steps = int(round(max_degrees / step))
    for n in range(-steps, steps + 1):
        angle = n * step
        rotated = ink.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        width = rotated.width
        data = rotated.tobytes()
        sums = [sum(data[r * width : (r + 1) * width]) for r in range(rotated.height)]
        mean = sum(sums) / len(sums)
        score = sum((s - mean) ** 2 for s in sums)
        if score > best_score:
            best_angle, best_score = angle, score
    return -best_angle


MIN_DESKEW_DEGREES = 1.0  # below this Tesseract copes better than a second resampling


def deskew(
    image: Image.Image, max_degrees: float = 3.0, min_degrees: float = MIN_DESKEW_DEGREES
) -> tuple[Image.Image, float]:
    """Straighten a page when it is skewed by at least ``min_degrees``.

    Returns the image and the angle removed (0.0 when left as is). Measured on the
    financial-statement scans: at 0.8 degrees rotating again costs accuracy (extra blur);
    at 1.5 degrees it recovers it.
    """
    angle = estimate_skew(image, max_degrees)
    if abs(angle) < min_degrees:
        return image, 0.0
    fill = 255 if image.mode in {"L", "1"} else (255, 255, 255)
    return image.rotate(-angle, resample=Image.Resampling.BICUBIC, fillcolor=fill), angle
