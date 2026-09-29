import logging
from functools import lru_cache
from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter, ImageOps


logger = logging.getLogger(__name__)
CONFIDENCE_THRESHOLDS = {1: 0.85, 2: 0.60, 3: 0.30}
# Radius is scaled by the longest image edge, then kept within practical bounds.
MASK_EXPANSION = {1: (0.002, 2, 6), 2: (0.005, 5, 10), 3: (0.008, 8, 16)}


class OcrEngineError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_engine():
    try:
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise OcrEngineError("OCR is unavailable. Install the project's OCR dependencies.") from error

    try:
        return PaddleOCR(
            ocr_version="PP-OCRv5",
            lang="korean",
            engine="onnxruntime",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as error:
        logger.exception("Could not initialize PaddleOCR")
        raise OcrEngineError("The local OCR engine could not be initialized.") from error


def detect_text_regions(contents: bytes, cleanup_level: int) -> list[dict]:
    try:
        import numpy as np

        with Image.open(BytesIO(contents)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image_array = np.asarray(image)[:, :, ::-1].copy()
        results = _get_engine().predict(image_array)
    except OcrEngineError:
        raise
    except Exception as error:
        logger.exception("PaddleOCR processing failed")
        raise OcrEngineError("Text detection failed. Please try another image.") from error

    threshold = CONFIDENCE_THRESHOLDS[cleanup_level]
    regions = []
    try:
        for result in results:
            payload = result.json
            if callable(payload):
                payload = payload()
            data = payload.get("res", payload)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polygons = data.get("rec_polys")
            if polygons is None:
                polygons = data.get("dt_polys", [])
            for text, score, polygon in zip(texts, scores, polygons):
                confidence = float(score)
                if confidence < threshold:
                    continue
                regions.append({
                    "text": str(text),
                    "confidence": confidence,
                    "polygon": [[float(point[0]), float(point[1])] for point in polygon],
                })
    except Exception as error:
        logger.exception("Could not parse PaddleOCR results")
        raise OcrEngineError("Text detection returned an unreadable result.") from error

    return regions


def generate_removal_mask(width: int, height: int, regions: list[dict], cleanup_level: int) -> bytes:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for region in regions:
        polygon = [(point[0], point[1]) for point in region["polygon"]]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=255)

    ratio, minimum, maximum = MASK_EXPANSION[cleanup_level]
    radius = max(minimum, min(maximum, round(max(width, height) * ratio)))
    if regions and radius:
        mask = mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))

    output = BytesIO()
    mask.save(output, format="PNG")
    return output.getvalue()
