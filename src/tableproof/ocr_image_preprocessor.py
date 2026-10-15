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
